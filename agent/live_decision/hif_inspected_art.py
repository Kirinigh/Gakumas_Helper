"""Retain independently read titles within a verified same-hand lifecycle.

These references establish a visual family only. They never establish upgrades,
effects, current resources or input freshness, and are scoped to one runtime.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np

try:
    from agent.card_selection.types import CandidateBox
except ModuleNotFoundError:
    from card_selection.types import CandidateBox

from .partial_card import build_live_card_views

INDEPENDENT_DETAIL_SCOPE = 'selected_full_title_returned_art_v1'
SELECTED_TITLE_SCOPE = 'selected_complete_title_dual_candidates_returned_art_v1'
LEFT_HALF_PROFILE = 'native_complete_left_half_v1'


def accepted_inspection_full_view(prediction, *, frame_id, family):
    """Reuse the existing full-view gates, not a new returned-art threshold.

    The agreement producer only accepts an independently accepted full-art
    result whose first family equals the masked result. Its original input
    profile floor is .70/.05; the ordinary masked proof remains .85/.05.
    """
    recognizer = prediction.get('recognizer')
    minimum = {'card_art_embedding_top5_hif_masked_v1': .85,
        'card_art_embedding_full_masked_agreement_v1': .70}.get(recognizer)
    return (minimum is not None and prediction.get('accepted') is True
        and prediction.get('reason') == 'accepted_visual_family'
        and prediction.get('view_mode') == 'full' and prediction.get('visible_ratio') == 1
        and prediction.get('frame_id') == frame_id and bool(family)
        and bool(prediction.get('top_k_card_ids'))
        and tuple(prediction['top_k_card_ids'][0]) == tuple(family)
        and type(prediction.get('confidence')) in (int, float)
        and minimum <= prediction['confidence'] <= 1
        and type(prediction.get('margin')) in (int, float)
        and .05 <= prediction['margin'] <= 1)


def selected_title_proof_valid(proof, *, targets, frame_ids, slot, family, catalog=None):
    """A complete catalog title resolves two agreeing candidates, not their scores."""
    from .upgrade import normalize_upgrade_title

    normalize = lambda text: normalize_upgrade_title(''.join(text.split()))
    try:
        detail = proof['detail']
        full = proof['full_predictions']
        masks = proof['masked_predictions']
        name = normalize(proof['catalog_name'])
        card_id = detail['card_id']
        if (proof.get('scope') != SELECTED_TITLE_SCOPE or proof.get('upgrade_verified') is not False
                or proof.get('slot') != slot or tuple(proof.get('family', ())) != tuple(family)
                or proof.get('plan') not in ('sense', 'logic', 'free')
                or not name or detail.get('accepted') is not True or card_id not in family
                or len(frame_ids) != 3 or len(set(frame_ids)) != 3
                or len(targets) != 3 or len(full) != 3 or len(masks) != 3 or len(detail['observations']) != 3):
            return False
        if catalog is not None:
            matching = {key for key, entry in catalog.entries.items()
                        if entry.plan in ('free', proof['plan']) and normalize(entry.name) == name}
            base_family = {key for key, entry in catalog.entries.items()
                           if entry.plan in ('free', proof['plan'])
                           and normalize(entry.name).rstrip('+') == name.rstrip('+')}
            if matching != {card_id} or base_family != set(family):
                return False
        for fid, target, masked, original, row in zip(frame_ids, targets, masks, full, detail['observations'], strict=True):
            if target.get('recognizer') == 'card_art_embedding_full_masked_agreement_v1':
                if (not accepted_inspection_full_view(target, frame_id=fid, family=family)
                        or {k:v for k,v in target.items() if k != 'recognizer'}
                            != {k:v for k,v in original.items() if k != 'recognizer'}):
                    return False
            elif target != masked:
                return False
            if (masked.get('recognizer') != 'card_art_embedding_top5_hif_masked_v1'
                    or original.get('recognizer') != 'card_art_embedding_top5_v1'
                    or tuple(masked.get('box', ())) != tuple(original.get('box', ()))
                    or tuple(masked.get('box', ()))[2:] != (190, 250)
                    or any(not masked.get(key) or masked[key] != original.get(key)
                           for key in ('model_sha256', 'classes_sha256'))):
                return False
            for prediction in (masked, original):
                if (prediction.get('frame_id') != fid or prediction.get('slot') != slot
                        or prediction.get('view_mode') != 'full' or prediction.get('visible_ratio') != 1
                        or type(prediction.get('accepted')) is not bool
                        or prediction.get('reason') not in ('accepted_visual_family',
                            'below_acceptance_threshold', 'below_margin_threshold')
                        or prediction['accepted'] != (prediction['reason'] == 'accepted_visual_family')
                        or not prediction.get('top_k_card_ids')
                        or tuple(prediction['top_k_card_ids'][0]) != tuple(family)
                        or any(type(prediction.get(key)) not in (int, float)
                               or not math.isfinite(prediction[key]) or not 0 <= prediction[key] <= 1
                               for key in ('confidence', 'margin'))):
                    return False
            roi, texts = row['roi'], row['ocr']
            if (row.get('frame_id') != fid or row.get('card_id') != card_id
                    or len(roi) != 4 or any(type(v) is not int for v in roi)
                    or min(roi[2:]) <= 0 or roi[0] < 0 or roi[1] < 360
                    or roi[0]+roi[2] > 720 or roi[1]+roi[3] > 1280 or len(texts) != 1):
                return False
            text = texts[0]
            box = text['box']
            if (type(text.get('score')) not in (int, float) or not .95 <= text['score'] <= 1
                    or normalize(text['text']) != name or len(box) != 4
                    or any(type(v) is not int for v in box) or min(box[2:]) <= 0
                    or min(box[:2]) < 0 or box[0]+box[2] > roi[2] or box[1]+box[3] > roi[3]):
                return False
        return True
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        return False


def selected_title_proof(frames, candidates, predictions, detail, recognizer, catalog, *,
                         selected_slot, plan, acceptance_threshold, min_margin, prepared_art=None):
    """Collect the unchanged full-art output only for a verified selected row."""
    from .partial_card import preprocess_live_card_view

    family = tuple(predictions[0][selected_slot].top_k_card_ids[0])
    card_id = detail.get('card_id')
    if card_id not in catalog.entries:
        return None
    full, masks = [], []
    for frame, cards, row in zip(frames, candidates, predictions, strict=True):
        profile = 'hif-four-card-art-v1' if len(cards) == 4 else 'hif-five-card-art-v2'
        view = build_live_card_views(frame.shape, cards, layout_profile=profile,
                                    selected_slot=selected_slot)[selected_slot]
        if view.visible_ratio != 1 or view.canvas_offset != (0, 0):
            return None
        tensor = preprocess_live_card_view(frame, view,
            source_color_order=recognizer.embedder.source_color_order)
        original = recognizer.classify_embedding(recognizer.embedder.embed_tensor(tensor),
            cards[selected_slot], frame_id=row[selected_slot].frame_id, plan=plan,
            acceptance_threshold=acceptance_threshold, min_margin=min_margin, resolve_upgrade_state=False)
        full.append(original.to_dict())
        target = row[selected_slot]
        if target.recognizer == 'card_art_embedding_full_masked_agreement_v1':
            if prepared_art is None or prepared_art.recognizer is not recognizer:
                return None
            # The combined result retains the full prediction, not the raw
            # masked prediction. Re-read that same fixed view explicitly.
            target = prepared_art.classify(frame, view, cards[selected_slot],
                frame_id=target.frame_id, plan=plan, acceptance_threshold=acceptance_threshold,
                min_margin=min_margin)
            target = replace(target, recognizer='card_art_embedding_top5_hif_masked_v1',
                             view_mode=view.view_mode, visible_ratio=view.visible_ratio)
        masks.append(target.to_dict())
    proof = dict(scope=SELECTED_TITLE_SCOPE, slot=selected_slot, family=family, plan=plan,
        catalog_name=catalog.entries[card_id].name, detail=detail, full_predictions=full, masked_predictions=masks,
        upgrade_verified=False)
    targets = [row[selected_slot].to_dict() for row in predictions]
    return proof if selected_title_proof_valid(proof, targets=targets,
        frame_ids=[p['frame_id'] for p in targets], slot=selected_slot, family=family, catalog=catalog) else None


def independent_detail_reference_valid(receipt, *, catalog=None):
    """Check the stronger receipt, without changing the legacy art threshold.

    The selected full view passed its original visual gate and independently
    supplied a complete title. All physical families/turns stayed unchanged;
    three actual returned crops must be identical. No expected deck is used.
    """
    from .upgrade import normalize_upgrade_title

    basis = receipt.get('independent_detail')
    if not isinstance(basis, dict) or basis.get('scope') not in (INDEPENDENT_DETAIL_SCOPE, SELECTED_TITLE_SCOPE):
        return False
    title_basis = basis['scope'] == SELECTED_TITLE_SCOPE
    profile = receipt.get('art', {}).get('visibility_profile')
    if profile not in (None, LEFT_HALF_PROFILE):
        return False
    if profile == LEFT_HALF_PROFILE or title_basis:
        cancel = basis.get('cancel', {})
        groups = [basis.get(name, {}).get('frame_ids', ()) for name in ('selected', 'before', 'after')]
        if ((profile == LEFT_HALF_PROFILE and not _left_half_art_valid(receipt['art'])) or not basis.get('source_run_id')
                or type(basis.get('source_revision')) is not int
                or any(len(ids) != 3 for ids in groups)
                or cancel.get('accepted') is not True or cancel.get('input_click_count') != 1
                or cancel.get('reason') != 'selection_cleared_without_business_submission'
                or tuple(cancel.get('target', ())) != (340, 300, 40, 40)
                or type(cancel.get('capture_age_ms')) not in (int, float)
                or not 0 <= cancel['capture_age_ms'] <= 1000):
            return False
        if title_basis and len(set(fid for ids in groups for fid in ids)) != 9:
            return False
    slot, family = receipt.get('slot'), tuple(receipt.get('family', ()))
    checks = [basis.get(name, {}) for name in ('selected', 'before', 'after')]
    detail = receipt.get('detail', {})
    selected_ids, returned_ids = receipt.get('selected_frame_ids', ()), receipt.get('reference_frame_ids', ())
    if (type(slot) is not int or not family or detail.get('accepted') is not True
            or detail.get('card_id') not in family
            or checks[0].get('frame_ids') != selected_ids or checks[2].get('frame_ids') != returned_ids):
        return False
    families = tuple(tuple(ids) for ids in checks[0].get('families', ()))
    turn = checks[0].get('turns_remaining')
    if len(families) not in (4, 5) or not 0 <= slot < len(families) or families[slot] != family or type(turn) is not int:
        return False
    if title_basis:
        from .selection import same_live_candidate_geometry, live_selection_geometry_evidence

        if (checks[0].get('selected_title_proof', {}).get('detail') != detail
                or checks[1].get('selected_title_proof', {}).get('detail', {}).get('card_id') != detail['card_id']):
            return False
        geometries = []
        try:
            for index, checkpoint in enumerate(checks):
                groups = [tuple(CandidateBox(p['slot'], tuple(p['box'])) for p in row['predictions'])
                          for row in checkpoint['identity_observations']]
                for group in groups:
                    geometry = live_selection_geometry_evidence(group)
                    if (len(group) != len(families) or tuple(c.slot for c in group) != tuple(range(len(families)))
                            or not geometry['available'] or geometry['ambiguous']
                            or tuple(geometry['raised_slots']) != (() if index == 2 else (slot,))
                            or not same_live_candidate_geometry(groups[0], group)):
                        return False
                geometries.append(groups[0])
            if not same_live_candidate_geometry(geometries[0], geometries[1]):
                return False
            returned_y = geometries[2][slot].box[1]
            if not 18 <= returned_y-geometries[1][slot].box[1] <= 34:
                return False
            lowered = tuple(replace(c, box=(c.box[0], returned_y, *c.box[2:])) if c.slot == slot else c
                            for c in geometries[1])
            if not same_live_candidate_geometry(lowered, geometries[2]):
                return False
        except (KeyError, IndexError, TypeError, ValueError):
            return False
    for index, checkpoint in enumerate(checks):
        frame_ids = checkpoint.get('frame_ids', ())
        selections, predictions = checkpoint.get('select_evidence', ()), checkpoint.get('identity_observations', ())
        expected_slot = None if index == 2 else slot
        if (len(frame_ids) != 3 or len(set(frame_ids)) != 3
                or checkpoint.get('selected_slot') != expected_slot
                or tuple(tuple(ids) for ids in checkpoint.get('families', ())) != families
                or checkpoint.get('turns_remaining') != turn or len(selections) != 3 or len(predictions) != 3):
            return False
        if title_basis and index < 2:
            if not selected_title_proof_valid(checkpoint.get('selected_title_proof', {}),
                    targets=[o.get('predictions', [])[slot] for o in predictions
                             if len(o.get('predictions', [])) > slot],
                    frame_ids=frame_ids, slot=slot, family=family, catalog=catalog):
                return False
        for fid, selection, observation in zip(frame_ids, selections, predictions, strict=True):
            slots = [] if expected_slot is None else [slot]
            if (selection.get('ocr_slots') != slots or selection.get('geometry_raised_slots') != slots
                    or selection.get('resolved_slots') != slots or selection.get('geometry_available') is not True
                    or selection.get('geometry_ambiguous') is not False or observation.get('frame_id') != fid):
                return False
            raw = observation.get('predictions', ())
            proofs = observation.get('title_proofs', ())
            neighbor_proofs = observation.get('neighbor_proofs', ())
            if (len(raw) != len(families) or [p.get('slot') for p in raw] != list(range(len(families)))
                    or any(p.get('frame_id') != fid for p in raw)
                    or len({p.get('slot') for p in proofs}) != len(proofs)
                    or any(p.get('slot') not in range(len(families)) for p in proofs)
                    or len(neighbor_proofs) > 1
                    or any(p.get('slot') != slot+1 for p in neighbor_proofs)):
                return False
            target = raw[slot]
            if (target.get('frame_id') != fid
                    or not target.get('top_k_card_ids') or tuple(target['top_k_card_ids'][0]) != family
                    or type(target.get('confidence')) not in (int,float) or not 0 <= target['confidence'] <= 1
                    or type(target.get('margin')) not in (int,float) or not 0 <= target['margin'] <= 1):
                return False
            if index < 2 and not title_basis and not accepted_inspection_full_view(target, frame_id=fid, family=family):
                return False
            if index == 2 and target.get('recognizer') != 'card_art_embedding_top5_hif_masked_v1':
                return False
            if profile == LEFT_HALF_PROFILE and index == 2:
                try:
                    cards = tuple(CandidateBox(slot=p['slot'], box=tuple(p['box'])) for p in raw)
                    layout = 'hif-four-card-art-v1' if len(cards) == 4 else 'hif-five-card-art-v2'
                    view = build_live_card_views((1280, 720, 3), cards, layout_profile=layout)[slot]
                    width = _left_half_width(view, slot, len(cards))
                    if receipt['art']['shape'] != [view.visible_box[3], width, 3]:
                        return False
                except (ValueError, KeyError, TypeError):
                    return False
            for i, prediction in enumerate(raw):
                if i == slot:
                    continue
                proof = next((p for p in proofs if p.get('slot') == i), None)
                neighbor = next((p for p in neighbor_proofs if p.get('slot') == i), None)
                if neighbor is not None:
                    from .hif_inspection_neighbor import valid_proof

                    if (proof is not None or expected_slot is None
                            or not valid_proof(neighbor, prediction=prediction, candidates=raw,
                                frame_id=fid, run_id=basis.get('source_run_id', receipt.get('run_id')),
                                revision=basis.get('source_revision', receipt.get('revision')),
                                selected_slot=slot, family=families[i], catalog=catalog)):
                        return False
                    continue
                if proof is None:
                    if (prediction.get('accepted') is not True or not prediction.get('top_k_card_ids')
                            or tuple(prediction['top_k_card_ids'][0]) != families[i]):
                        return False
                elif (proof.get('accepted') is not True or proof.get('frame_id') != fid
                        or tuple(proof.get('family', ())) != families[i] or proof.get('upgrade_verified') is not False
                        or type(proof.get('confidence')) not in (int,float) or not .95 <= proof['confidence'] <= 1):
                    return False
                if proof is not None:
                    roi, box = proof.get('roi', ()), prediction.get('box', ())
                    if (len(roi) != 4 or len(box) != 4 or any(type(v) is not int for v in (*roi,*box))
                            or min(roi[2:]) <= 0 or min(box[2:]) <= 0
                            or roi[0] < box[0] or roi[1] < box[1]
                            or roi[0]+roi[2] > box[0]+box[2] or roi[1]+roi[3] > box[1]+box[3]):
                        return False
                if proof is not None and catalog is not None:
                    from types import SimpleNamespace

                    from .hif_bound_title import in_candidate_band
                    from .hif_inspection_neighbor import _source_identity_valid

                    text = normalize_upgrade_title(''.join(proof.get('text', '').split())).rstrip('+')
                    matching = {key for key, entry in catalog.entries.items()
                        if normalize_upgrade_title(''.join(entry.name.split())).rstrip('+') == text}
                    raw_prediction = SimpleNamespace(**prediction)
                    # The complete-title producer has two existing routes:
                    # weak confidence with an unambiguous top family, or a
                    # low-margin candidate inside its bounded top-five band.
                    # Requiring only the latter rejects valid high-margin
                    # title evidence when restoring an inspected reference.
                    title_source = _source_identity_valid(raw_prediction,
                        frame_id=fid, family=families[i], title_proof=proof, catalog=catalog)
                    if matching != set(families[i]) or not (
                            title_source or in_candidate_band(raw_prediction, matching, .05)):
                        return False
    rows = detail.get('observations', ())
    if len(rows) != 3 or [row.get('frame_id') for row in rows] != list(selected_ids):
        return False
    names = []
    for row in rows:
        roi, texts = row.get('roi', ()), row.get('ocr', ())
        valid = [text for text in texts if type(text.get('score')) in (int,float) and .95 <= text['score'] <= 1
            and len(text.get('box', ())) == 4 and len(roi) == 4 and text['box'][0] >= 0 and text['box'][1] >= 0
            and min(roi[2:]) > 0 and min(text['box'][2:]) > 0
            and text['box'][0]+text['box'][2] <= roi[2] and text['box'][1]+text['box'][3] <= roi[3]]
        if row.get('card_id') != detail['card_id'] or len(valid) != 1:
            return False
        names.append(normalize_upgrade_title(''.join(valid[0].get('text', '').split())))
    if not names[0] or len(set(names)) != 1:
        return False
    if catalog is not None and (detail['card_id'] not in catalog.entries or names[0] != normalize_upgrade_title(
            ''.join(catalog.entries[detail['card_id']].name.split()))):
        return False
    arts = basis.get('returned_art', ())
    return (len(arts) == 3 and [row.get('frame_id') for row in arts] == list(returned_ids)
        and all(row.get('art') == receipt.get('art') for row in arts))


def circle_interior(art):
    """Exclude the circular portrait's transparent rim, never compare the stage."""
    pixels = np.frombuffer(bytes.fromhex(art['pixels']), dtype=np.uint8).reshape(art['shape']).copy()
    height, width = pixels.shape[:2]
    yy, xx = np.ogrid[:height, :width]
    center = (height-1)/2
    # The calibrated art square encloses the circular portrait. Its two-pixel
    # antialiased edge blends with the animated stage; only the interior is art.
    inside = ((xx+art['offset'][0]-center)**2+(yy-center)**2 <= (center-2)**2)
    pixels[~inside] = 0
    return dict(art, pixels=pixels.tobytes().hex(), mask='circular_interior_v1')


def visible_art(image, candidates, slot, *, selected_slot=None, visibility_profile=None):
    count = len(candidates)
    if count not in (4, 5):
        raise ValueError('inspected art requires an overlapping unselected hand')
    profile = 'hif-four-card-art-v1' if count == 4 else 'hif-five-card-art-v2'
    view = build_live_card_views(image.shape, candidates, layout_profile=profile,
                                selected_slot=selected_slot)[slot]
    x, y, width, height = view.visible_box
    extra = {}
    if visibility_profile == LEFT_HALF_PROFILE:
        # A fixed native half, never a width chosen from observed differences.
        # Require its entire circular interior to be visible. In particular,
        # do not intersect it with a narrower crop or a left-occluded neighbor.
        width = _left_half_width(view, slot, count)
        extra = dict(visibility_profile=LEFT_HALF_PROFILE, canvas_size=[height, height])
    elif visibility_profile is not None:
        raise ValueError('inspected artwork visibility profile is unknown')
    # Exact equality is stricter than another similarity threshold/embedding.
    crop = image[y:y+height, x:x+width]
    return circle_interior(dict(shape=list(crop.shape), pixels=crop.tobytes().hex(),
                layout=profile, offset=list(view.canvas_offset), **extra))


def _left_half_width(view, slot, count):
    width, height = view.visible_box[2:]
    if (slot >= count-1 or view.canvas_size != (height, height)
            or view.canvas_offset != (0, 0) or width >= height
            or width < (height+1)//2):
        raise ValueError('inspected left half is not fully visible with a right neighbor')
    return (height+1)//2


def _left_half_art_valid(art):
    shape = art.get('shape', ())
    if (len(shape) != 3 or any(type(n) is not int for n in shape)
            or not 176 <= shape[0] <= 192 or shape[1:] != [(shape[0]+1)//2, 3]
            or art.get('canvas_size') != [shape[0], shape[0]] or art.get('offset') != [0, 0]
            or art.get('layout') not in ('hif-four-card-art-v1', 'hif-five-card-art-v2')
            or art.get('mask') != 'circular_interior_v1'):
        return False
    try:
        return len(bytes.fromhex(art['pixels'])) == math.prod(shape) and circle_interior(art) == art
    except (ValueError, KeyError, TypeError):
        return False


def _same_inspected_hand(context, receipt, candidates, predictions, *, selected_slot=None):
    """An already read identity lasts through free observation of the same hand."""
    from .selection import same_live_candidate_geometry

    after = receipt.get('independent_detail', {}).get('after', {})
    families = tuple(tuple(family) for family in after.get('families', ()))
    if (not families or len(predictions) != len(candidates) or len(predictions) != len(families)
            or any(p.slot != i or c.slot != i or tuple(p.box) != tuple(c.box)
                or not p.top_k_card_ids or (p.accepted or i != receipt['slot'])
                and tuple(p.top_k_card_ids[0]) != families[i]
                for i, (c, p) in enumerate(zip(candidates, predictions, strict=True)))):
        return None
    original = after.get('identity_observations', ())
    # Selecting a different card raises that card. Its current selection is
    # independently checked by the caller; compare the unchanged slots only.
    if not original or not same_live_candidate_geometry(tuple(
            CandidateBox(p['slot'], tuple(p['box'])) for p in original[0]['predictions']
            if p['slot'] != selected_slot), tuple(c for c in candidates if c.slot != selected_slot)):
        return None
    active = getattr(context, 'gkh_hif_current_hand_inspection', None)
    if (isinstance(active, dict) and active.get('accepted') is True
            and active.get('business_submit_count') == 0
            and active.get('inspected_art_reference') == receipt):
        return 'no_business_input_since_inspection'
    from .hif_bound_title import binding

    known = binding(context, receipt['slot'])
    bundle = getattr(context, 'gkh_hif_opening_state', None)
    baseline = getattr(context, 'gkh_hif_committed_event', {})
    if (known is not None and known['canonical_card_id'] in receipt['family']
            and bundle.state.turns_remaining == after.get('turns_remaining')
            and tuple(tuple(f) for f in baseline.get('core', {}).get('families', ())) == families):
        return 'verified_same_hand_business_state'
    return None


def inspected_family(prediction, proof):
    """Separate a verified identity from the unchanged weak model ranking."""
    if (not isinstance(proof, dict) or proof.get('continuity_scope') not in (
            'no_business_input_since_inspection', 'verified_same_hand_business_state')
            or proof.get('upgrade_verified') is not False or not prediction.accepted
            or proof.get('slot') != prediction.slot or proof.get('frame_id') != prediction.frame_id):
        return ()
    raw, current = proof.get('original_prediction', {}), prediction.to_dict()
    if raw.get('accepted') is not False and not (
            raw.get('accepted') is True and raw.get('reason') == 'accepted_visual_family'
            and prediction.reason == 'upstream_inspected_complete_title'
            and proof.get('scope') == 'upstream_inspected_complete_title'
            and proof.get('detail', {}).get('accepted') is True
            and proof['detail'].get('card_id') in proof.get('family', ())):
        return ()
    # Later enhancement resolution can change card_id, reason and confidence.
    # The model's original ranking, scores, frame and geometry stay intact.
    keys = ('slot', 'box', 'frame_id', 'margin', 'raw_score', 'model_sha256',
            'classes_sha256', 'gallery_sha256', 'top_k_card_ids', 'top_k_scores')
    if any(json.dumps(raw.get(k)) != json.dumps(current.get(k)) for k in keys):
        return ()
    return tuple(proof.get('family', ()))


def corroborate(context, image, candidates, predictions, *, selected_slot, evidence):
    from .hif_opening_art import corroborate as opening_corroboration

    predictions = opening_corroboration(context, image, candidates, predictions,
        selected_slot=selected_slot, evidence=evidence)
    run = getattr(context, 'gkh_hif_live_context', None)
    receipt = getattr(context, 'gkh_hif_inspected_art', None)
    if (run is None or not receipt
            or (receipt.get('run_id'), receipt.get('revision')) != (run.run_id, run.revision)):
        return tuple(predictions)
    slot = receipt['slot']
    if (not 0 <= slot < len(predictions)
            or (selected_slot is not None and slot in (selected_slot, selected_slot+1))):
        return tuple(predictions)
    prediction = predictions[slot]
    profile = receipt.get('art', {}).get('visibility_profile')
    if profile not in (None, LEFT_HALF_PROFILE):
        return tuple(predictions)
    independent = independent_detail_reference_valid(receipt)
    if profile == LEFT_HALF_PROFILE and not independent:
        return tuple(predictions)
    legacy = prediction.reason == 'below_margin_threshold' and prediction.confidence >= .85
    if (prediction.accepted or not (legacy or independent and prediction.reason in (
            'below_acceptance_threshold', 'below_margin_threshold'))
            or prediction.recognizer != 'card_art_embedding_top5_hif_masked_v1'
            or not math.isfinite(prediction.confidence) or not 0 <= prediction.confidence <= 1
            or not math.isfinite(prediction.margin) or not 0 <= prediction.margin <= 1
            or not prediction.top_k_card_ids):
        return tuple(predictions)
    continuity = _same_inspected_hand(context, receipt, candidates, predictions,
        selected_slot=selected_slot) if independent else None
    if continuity is None:
        if tuple(prediction.top_k_card_ids[0]) != tuple(receipt['family']):
            return tuple(predictions)
        try:
            current_art = visible_art(image, candidates, slot, selected_slot=selected_slot,
                                      visibility_profile=profile)
        except ValueError:
            return tuple(predictions)
        if current_art != receipt['art']:
            return tuple(predictions)
    resolved = list(predictions)
    resolved[slot] = replace(prediction, accepted=True,
        confidence=receipt['title_confidence'],
        reason='inspected_title_same_hand' if continuity else 'inspected_title_exact_visible_art')
    evidence.append(dict(slot=slot, family=receipt['family'],
        selected_frame_ids=receipt['selected_frame_ids'],
        reference_frame_ids=receipt['reference_frame_ids'],
        frame_id=prediction.frame_id, upgrade_verified=False,
        original_prediction=prediction.to_dict(),
        **(dict(continuity_scope=continuity) if continuity else {})))
    return tuple(resolved)


def _observation_budget_valid(value, attempt):
    return (isinstance(value, dict)
        and set(value) == {'observation_stage', 'observation_attempt', 'elapsed_ms',
                          'max_observation_attempts', 'observation_budget_ms'}
        and value.get('observation_stage') == 'after_clear'
        and type(value.get('observation_attempt')) is int and value['observation_attempt'] == attempt
        and type(value.get('max_observation_attempts')) is int and value['max_observation_attempts'] == 8
        and value.get('observation_budget_ms') == 10000
        and type(value.get('elapsed_ms')) in (int, float) and 0 <= value['elapsed_ms'] < 10000)


def _three_frame_ids(ids):
    return (isinstance(ids, (tuple, list)) and len(ids) == 3
        and all(isinstance(fid, str) and fid for fid in ids) and len(set(ids)) == 3)


def restore_reference(source, run, catalog):
    """Rebind a completed observation-only reset; callers verify game process."""
    from .upgrade import normalize_upgrade_title

    receipt = source.get('inspected_art_reference')
    events = source.get('pending_inspection_reset', ())
    if not isinstance(receipt, dict) or not 1 <= len(events) <= 8:
        raise ValueError('inspected family source has no complete reset')
    event = events[0]
    final_budget = event.get('after', {}).get('observation_budget')
    bounded = final_budget is not None or any('observation_stage' in retry for retry in events[1:])
    if bounded and not _observation_budget_valid(final_budget, len(events)):
        raise ValueError('inspected family final observation budget is incomplete')
    if not bounded and len(events) > 2:
        raise ValueError('inspected family legacy source has too many retries')
    elapsed = 0
    empty_hand_retries = 0
    # The cancel precedes all after observations. These rejected groups only
    # account for scheduling; the final reference still needs its strong proof.
    for attempt, retry in enumerate(events[1:], start=1):
        evidence = retry.get('evidence', ()) if isinstance(retry, dict) else ()
        detail = evidence[0] if isinstance(evidence, (tuple, list)) and len(evidence) == 1 else {}
        frame_ids = detail.get('frame_ids', ()) if isinstance(detail, dict) else ()
        reason = retry.get('reason') if isinstance(retry, dict) else None
        retry_keys = {'phase', 'reason', 'input_click_count', 'evidence'}
        if bounded:
            budget_keys = set(final_budget)
            retry_keys |= budget_keys | {'observation_frame_ids'}
            budget = {key: retry.get(key) for key in budget_keys}
            observed_ids = retry.get('observation_frame_ids')
            if (not _observation_budget_valid(budget, attempt)
                    or not elapsed <= budget['elapsed_ms'] <= final_budget['elapsed_ms']
                    or not _three_frame_ids(observed_ids)):
                raise ValueError('inspected family after observation sequence is invalid')
            elapsed = budget['elapsed_ms']
            if frame_ids and list(frame_ids) != list(observed_ids):
                raise ValueError('inspected family retry frames differ from the captured group')
            frame_ids = observed_ids
        detail_keys = {'reason', 'frame_ids', 'input_click_count'}
        if reason == 'inspected_target_weak_identity':
            detail_keys |= {'inspect_slot', 'selected_slot', 'candidate_boxes',
                            'turns_remaining', 'identity_observations'}
        elif bounded and reason == 'exact_select_low_confidence':
            detail_keys = {'phase', 'reason', 'slot', 'confidence', 'frame_id', 'input_click_count'}
            if (detail.get('phase') != 'hif_select_label' or detail.get('frame_id') not in frame_ids
                    or type(detail.get('confidence')) not in (int, float)
                    or not 0 <= detail['confidence'] < .95):
                raise ValueError('inspected family SELECT retry is incomplete')
        elif bounded and reason == 'single_frame_hand_count_outlier':
            detail_keys = {'reason', 'frame_ids', 'candidate_boxes'}
        elif bounded and reason == 'inspected_single_empty_hand_frame':
            detail_keys = {'reason', 'frame_ids', 'candidate_boxes', 'complete_select_evidence'}
        complete_title_retry = (reason == 'inspected_single_empty_hand_frame'
            and receipt.get('independent_detail', {}).get('scope') == SELECTED_TITLE_SCOPE)
        if (receipt.get('art', {}).get('visibility_profile') != LEFT_HALF_PROFILE and not complete_title_retry
                or event.get('phase') != 'selection_reset'
                or not isinstance(retry, dict)
                or set(retry) != retry_keys
                or retry.get('phase') != 'selection_reset_observation_retry'
                or reason not in (('inspected_left_half_pixel_unstable', 'inspected_target_weak_identity',
                                  'exact_select_low_confidence', 'single_frame_hand_count_outlier',
                                  'inspected_single_empty_hand_frame') if bounded
                                 else ('inspected_left_half_pixel_unstable', 'inspected_target_weak_identity'))
                or type(retry.get('input_click_count')) is not int or retry['input_click_count'] != 0
                or not isinstance(detail, dict) or set(detail) != detail_keys
                or detail.get('reason') != retry['reason']
                or type(detail.get('input_click_count', 0)) is not int or detail.get('input_click_count', 0) != 0
                or not _three_frame_ids(frame_ids)):
            raise ValueError('inspected family source has an invalid after-cancel retry')
        if reason == 'inspected_single_empty_hand_frame':
            from .selection import same_live_candidate_geometry, live_selection_geometry_evidence

            empty_hand_retries += 1
            boxes = detail['candidate_boxes']
            proofs = detail['complete_select_evidence']
            unselected = dict(ocr_slots=[], geometry_available=True, geometry_ambiguous=False,
                geometry_raised_slots=[], geometry_slot=None, resolved_slots=[], used_geometry_fallback=False)
            if (empty_hand_retries != 1 or len(event.get('before', {}).get('families', ())) != 5
                    or not isinstance(boxes, (tuple, list)) or len(boxes) != 3
                    or any(not isinstance(row, (tuple, list)) for row in boxes)
                    or sorted(map(len, boxes)) != [0, 5, 5]
                    or not isinstance(proofs, (tuple, list)) or len(proofs) != 2
                    or any(proof != unselected for proof in proofs)
                    or any(len(box) != 4 or any(type(v) is not int for v in box)
                           or box[2] <= 0 or box[3] <= 0 for row in boxes for box in row)):
                raise ValueError('inspected empty-hand retry proof is incomplete or repeated')
            complete = [tuple(CandidateBox(i, tuple(box)) for i, box in enumerate(row)) for row in boxes if row]
            if (not same_live_candidate_geometry(*complete)
                    or any(live_selection_geometry_evidence(row) != dict(available=True, ambiguous=False,
                        raised_slots=(), selected_slot=None) for row in complete)
                    or any(not same_live_candidate_geometry(complete[0], tuple(
                        CandidateBox(i, tuple(p['box'])) for i, p in enumerate(row['predictions'])))
                        for row in event.get('after', {}).get('identity_observations', ()))):
                raise ValueError('inspected empty-hand retry geometry changed')
        if reason == 'inspected_target_weak_identity':
            # This rejected batch only accounts for zero-input scheduling. Do
            # not use its predictions as identity evidence for the final art.
            count = len(event.get('before', {}).get('families', ()))
            rows = detail['identity_observations']
            boxes = detail['candidate_boxes']
            if (detail['inspect_slot'] != receipt.get('slot') or detail['selected_slot'] is not None
                    or detail['turns_remaining'] != [event.get('before', {}).get('turns_remaining')] * 3
                    or len(rows) != 3 or [row.get('frame_id') for row in rows] != list(frame_ids)
                    or len(boxes) != 3 or any(len(group) != count for group in boxes)
                    or any(len(row.get('predictions', ())) != count for row in rows)):
                raise ValueError('inspected weak-target retry differs from the completed cancellation')
    before, after = event.get('before', {}), event.get('after', {})
    original_run = source.get('execution_live_context', source.get('observed_live_context', {}))
    slot = receipt.get('slot')
    if (type(slot) is not int or not 0 <= slot < len(after.get('families', ()))
            or len(after['families']) not in (4, 5)
            or type(receipt.get('title_confidence')) not in (int, float)
            or not .95 <= receipt['title_confidence'] <= 1):
        raise ValueError('inspected family slot or confidence is invalid')
    if (event.get('accepted') is not True or event.get('input_click_count') != 1
            or event.get('reason') != 'selection_cleared_without_business_submission'
            or event.get('inspected_family') != {k:v for k,v in receipt.items() if k != 'art'}
            or before.get('selected_slot') != receipt.get('slot') or after.get('selected_slot') is not None
            or any(before.get(k) != after.get(k) for k in ('families', 'turns_remaining'))
            or receipt.get('family') != after['families'][slot]
            or receipt.get('reference_frame_ids') != after.get('frame_ids')
            or any(original_run.get(k) != receipt.get(k) for k in ('run_id', 'revision'))
            or any(original_run.get(k) != getattr(run, k) for k in ('idol', 'plan', 'max_stamina'))):
        raise ValueError('inspected family reset continuity is invalid')
    detail = receipt.get('detail', {})
    card_id = detail.get('card_id')
    observations = detail.get('observations', ())
    if (detail.get('accepted') is not True or card_id not in receipt['family']
            or card_id not in catalog.entries or len(observations) != 3
            or len(set(receipt['selected_frame_ids'])) != 3
            or len(set(receipt['reference_frame_ids'])) != 3
            or [row.get('frame_id') for row in observations] != receipt['selected_frame_ids']):
        raise ValueError('inspected family full-title evidence is incomplete')
    title = normalize_upgrade_title(''.join(catalog.entries[card_id].name.split()))
    for observation in observations:
        matches = [row for row in observation.get('ocr', ())
            if .95 <= row.get('score', 0) <= 1
            and normalize_upgrade_title(''.join(row.get('text', '').split())) == title]
        if observation.get('card_id') != card_id or len(matches) != 1:
            raise ValueError('inspected family full title no longer matches the catalog')
    art = receipt.get('art', {})
    profile = art.get('visibility_profile')
    if (profile not in (None, LEFT_HALF_PROFILE)
            or profile == LEFT_HALF_PROFILE and 'independent_detail' not in receipt):
        raise ValueError('inspected artwork visibility profile has no independent basis')
    shape = art.get('shape', ())
    if (len(shape) != 3 or shape[2] != 3 or not 8 <= shape[1] <= 192
            or not 176 <= shape[0] <= 192
            or len(bytes.fromhex(art.get('pixels', ''))) != math.prod(shape)):
        raise ValueError('inspected family reference pixels are incomplete')
    if 'independent_detail' in receipt:
        basis = receipt['independent_detail']
        if (not independent_detail_reference_valid(receipt, catalog=catalog)
                or basis['before'] != before or basis['after'] != after):
            raise ValueError('independent inspected detail continuity is incomplete')
        if (profile == LEFT_HALF_PROFILE or basis.get('scope') == SELECTED_TITLE_SCOPE) and (
                (basis['source_run_id'], basis['source_revision']) != (receipt['run_id'], receipt['revision'])
                or basis['cancel'] != {k: event[k] for k in basis['cancel']}):
            raise ValueError('inspected left half cancellation source is inconsistent')
    return dict(receipt, run_id=run.run_id, revision=run.revision, art=circle_interior(art))


def source_title_evidence_valid(context, certificate):
    """Validate a prior inspected observation as source identity, never new input."""
    receipt = getattr(context, 'gkh_hif_inspected_art', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    source = certificate.get('source_prediction')
    proof = certificate.get('inspected_art_evidence')
    if (not isinstance(receipt, dict) or run is None or source is None
            or not callable(getattr(source, 'to_dict', None)) or not isinstance(proof, dict)
            or (receipt.get('run_id'), receipt.get('revision')) != (run.run_id, run.revision)
            or certificate.get('slot') != receipt.get('slot') or source.slot != receipt.get('slot')
            or tuple(certificate.get('family', ())) != tuple(receipt.get('family', ()))
            or certificate.get('card_id') not in receipt.get('family', ())
            or source.accepted is not True or source.card_id != certificate.get('card_id')
            or type(receipt.get('title_confidence')) not in (int, float)
            or not .95 <= receipt['title_confidence'] <= 1
            or source.confidence != receipt['title_confidence']
            or certificate.get('probe_confidence') != source.confidence
            or proof.get('slot') != source.slot or proof.get('frame_id') != source.frame_id
            or tuple(proof.get('family', ())) != tuple(receipt.get('family', ()))
            or proof.get('selected_frame_ids') != receipt.get('selected_frame_ids')
            or proof.get('reference_frame_ids') != receipt.get('reference_frame_ids')
            or proof.get('upgrade_verified') is not False
            or not independent_detail_reference_valid(receipt)):
        return False
    raw = proof.get('original_prediction', {})
    actual = source.to_dict()
    # Serialized certificates turn tuples into lists; only container spelling
    # may differ, never a rank, coordinate or score.
    try:
        for name in ('box', 'top_k_scores'):
            if tuple(raw[name]) != tuple(actual[name]):
                return False
        if tuple(map(tuple,raw['top_k_card_ids'])) != tuple(map(tuple,actual['top_k_card_ids'])):
            return False
    except (KeyError, TypeError):
        return False
    return (raw.get('accepted') is False
        and raw.get('recognizer') == 'card_art_embedding_top5_hif_masked_v1'
        and raw.get('reason') in ('below_acceptance_threshold', 'below_margin_threshold')
        and (tuple((raw.get('top_k_card_ids') or [()])[0]) == tuple(receipt['family'])
             or inspected_family(source, proof) == tuple(receipt['family']))
        and all(raw.get(key) == actual.get(key) for key in (
            'slot', 'frame_id', 'margin', 'raw_score', 'model_sha256',
            'classes_sha256', 'gallery_sha256')))


def margin_evidence(context, certificate, fresh, event, minimum_margin):
    """Keep embedding margins unchanged and explicitly qualify the other proof."""
    receipt = getattr(context, 'gkh_hif_inspected_art', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    if (not receipt or run is None
            or (receipt['run_id'], receipt['revision']) != (run.run_id, run.revision)
            or certificate['slot'] != receipt['slot']
            or tuple(certificate['family']) != tuple(receipt['family'])
            or certificate['card_id'] not in receipt['family']):
        return None

    def agrees(proof, frame_id):
        return (isinstance(proof, dict) and proof.get('slot') == receipt['slot']
            and tuple(proof.get('family', ())) == tuple(receipt['family'])
            and proof.get('frame_id') == frame_id
            and proof.get('selected_frame_ids') == receipt['selected_frame_ids']
            and proof.get('reference_frame_ids') == receipt['reference_frame_ids']
            and proof.get('upgrade_verified') is False)

    probe = certificate.get('inspected_art_evidence')
    probe_ok = (certificate['probe_margin'] >= minimum_margin or agrees(probe,
        certificate['source_prediction'].frame_id))
    current = [p for p in event.get('inspected_title_art_families', ()) if agrees(p, fresh.frame_id)]
    fresh_ok = (fresh.accepted and fresh.margin >= minimum_margin or fresh.accepted
        and fresh.reason in ('inspected_title_exact_visible_art', 'inspected_title_same_hand') and len(current) == 1)
    if not probe_ok or not fresh_ok:
        return None
    return dict(slot=certificate['slot'], probe_margin=certificate['probe_margin'],
        fresh_margin=fresh.margin, probe_uses_inspected_art=certificate['probe_margin'] < minimum_margin,
        fresh_uses_inspected_art=fresh.margin < minimum_margin, upgrade_verified=False)
