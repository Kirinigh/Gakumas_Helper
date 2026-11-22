"""Current complete-title family evidence for an already verified HIF hand.

The original embedding ranking and margin remain untouched. This proof neither
establishes enhancement nor replaces selection, resource or input-age checks.
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace
from dataclasses import asdict

SCOPE = 'bound_current_title_family'


def full_view_art_evidence_valid(evidence, prediction, family_ids, minimum_margin):
    """Validate independent full artwork; the caller must separately prove the title."""
    if not isinstance(evidence, dict):
        return False
    full = evidence.get('prediction', {})
    groups = full.get('top_k_card_ids', ())
    boxes = evidence.get('candidate_boxes', ())
    return (evidence.get('scope') == 'complete_row_full_art_v1'
        and prediction.accepted is False
        and prediction.recognizer == 'card_art_embedding_top5_hif_masked_v1'
        and prediction.reason in ('below_acceptance_threshold', 'below_margin_threshold')
        and prediction.view_mode == 'full' and prediction.visible_ratio == 1
        and tuple(prediction.box[2:]) == (190, 250)
        and len(boxes) in (4, 5) and 0 <= prediction.slot < len(boxes)
        and tuple(boxes[prediction.slot]) == tuple(prediction.box)
        and evidence.get('selected_slot') in (None, prediction.slot)
        and full.get('recognizer') == 'card_art_embedding_top5_v1'
        and full.get('accepted') is True and full.get('reason') == 'accepted_visual_family'
        and full.get('slot') == prediction.slot and full.get('frame_id') == prediction.frame_id
        and tuple(full.get('box', ())) == tuple(prediction.box)
        and bool(family_ids) and bool(groups) and set(groups[0]) == set(family_ids)
        and all(full.get(key) and full[key] == getattr(prediction, key)
                for key in ('model_sha256', 'classes_sha256'))
        and type(full.get('confidence')) in (int, float) and .70 <= full['confidence'] <= 1
        and type(full.get('margin')) in (int, float)
        and max(.05, minimum_margin) <= full['margin'] <= 1)


def full_view_art_evidence(context, frame, candidates, prediction, family_ids, *,
                           minimum_margin, selected_slot=None):
    """Read a fully exposed card in an independently established current row.

    The mask may remove the distinguishing artwork. A full catalog title plus
    the existing .70/.05 full-art gate can establish its family without changing
    the masked ranking, enhancement, or any input authorization.
    """
    from .hif_visual import hif_selected_row_proposal, hif_unselected_row_candidates
    from .partial_card import build_live_card_views, preprocess_live_card_view
    try:
        from agent.card_selection.model import frame_identifier
    except ModuleNotFoundError:
        from card_selection.model import frame_identifier

    prepared = getattr(context, 'gkh_hif_card_art', None)
    recognizer = getattr(prepared, 'recognizer', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    if (recognizer is None or run is None or len(candidates) not in (4, 5)
            or prediction.accepted or prediction.frame_id != frame_identifier(frame)
            or not 0 <= prediction.slot < len(candidates)
            or tuple(candidates[prediction.slot].box) != tuple(prediction.box)):
        return None
    if selected_slot is None:
        row = hif_unselected_row_candidates(frame.shape, candidates, image=frame)
    else:
        proposal = hif_selected_row_proposal(frame, candidates, hand_count=len(candidates))
        row = proposal[0] if proposal is not None and proposal[1] == selected_slot else None
    if row is None or tuple(c.box for c in row) != tuple(c.box for c in candidates):
        return None
    profile = 'hif-four-card-art-v1' if len(candidates) == 4 else 'hif-five-card-art-v2'
    view = build_live_card_views(frame.shape, candidates, layout_profile=profile,
                                selected_slot=selected_slot)[prediction.slot]
    if view.visible_ratio != 1 or view.canvas_offset != (0, 0):
        return None
    tensor = preprocess_live_card_view(frame, view,
        source_color_order=recognizer.embedder.source_color_order)
    full = recognizer.classify_embedding(recognizer.embedder.embed_tensor(tensor),
        candidates[prediction.slot], frame_id=prediction.frame_id, plan=run.plan,
        acceptance_threshold=.70, min_margin=max(.05, minimum_margin), resolve_upgrade_state=False)
    evidence = dict(scope='complete_row_full_art_v1', prediction=full.to_dict(),
        candidate_boxes=[list(c.box) for c in candidates], selected_slot=selected_slot)
    return evidence if full_view_art_evidence_valid(evidence, prediction, family_ids, minimum_margin) else None


def _fresh_opening(context, bundle, run):
    """Use the native opening as identity evidence, never as a settlement."""
    event = getattr(context, 'gkh_hif_complete_opening_event', None)
    if (getattr(context, 'gkh_hif_committed_state', None) is not None
            or not isinstance(event, dict) or event.get('event') != 'live_opening_state'
            or event.get('accepted') is not True or event.get('strategy_state_bound') is not True
            or event.get('executed') is not False or event.get('execution_submit_count') != 0
            or (event.get('run_id'), event.get('revision')) != (run.run_id, run.revision)
            or bundle.state.turn != 1):
        return None
    components = event.get('components', {})
    if any(not isinstance(components.get(key), dict)
           or components[key].get('accepted') is not True
           or components[key].get('returned_unchanged') is not True
           or (components[key].get('run_id'), components[key].get('revision'))
               != (run.run_id, run.revision) for key in ('cards', 'status', 'items', 'drinks', 'turns')):
        return None
    # A separately verified current-turn support projection is not a business
    # transition. All remaining opening state must still match exactly.
    observed = event.get('state')
    if not isinstance(observed, dict):
        return None
    current = asdict(bundle.state)
    if json.dumps({k: v for k, v in current.items() if k != 'hand_support'}, sort_keys=True) != json.dumps(
            {k: v for k, v in observed.items() if k != 'hand_support'}, sort_keys=True):
        return None
    return event


def binding(context, slot):
    bundle = getattr(context, 'gkh_hif_opening_state', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    event = getattr(context, 'gkh_hif_committed_event', None)
    if (bundle is None or run is None
            or (bundle.run_id, bundle.revision) != (run.run_id, run.revision)):
        return None
    fresh = event is None
    if fresh:
        event = _fresh_opening(context, bundle, run)
        if event is None:
            return None
    elif (not isinstance(event, dict)
          or (event.get('run_id'), event.get('revision')) != (run.run_id, run.revision)
          or event.get('accepted') is not True or not event.get('checks')
          or not all(value is True for value in event['checks'].values())):
        return None
    hand = bundle.state.deck.hand
    core = event.get('core', {})
    families = core.get('families', ())
    ids = tuple(ref.instance_id for ref in hand)
    if (not 0 <= slot < len(hand) or len(set(ids)) != len(ids)
            or len(families) != len(hand) or len(set(core.get('frame_ids', ()))) != 3
            or (not fresh and tuple(event.get('observed_hand_instance_ids', ())) != ids)):
        return None
    keys = []
    for ref, family_ids in zip(hand, families, strict=True):
        definition = bundle.kernel.cards.get(ref.card_id)
        if definition is None:
            return None
        canonical = definition.canonical_card_id or ref.card_id
        matches = [key for key in bundle.inventory_keys if key[0] == ref.instance_id]
        if len(matches) != 1 or matches[0][1] != canonical or canonical not in family_ids:
            return None
        keys.append(matches[0])
    return dict(run_id=run.run_id, revision=run.revision, slot=slot,
        card_instance_id=ids[slot], canonical_card_id=keys[slot][1],
        hand_instance_ids=ids, inventory_key=keys[slot])


def inventory_families(context):
    """Retain complete deck identities for navigation, without binding slots.

    A post-draw hand has no settled physical order yet. A fresh full title can
    corroborate an artwork candidate from this inventory, but cannot identify
    its instance, enhancement or authorize an input.
    """
    bundle = getattr(context, 'gkh_hif_opening_state', None)
    run = getattr(context, 'gkh_hif_live_context', None)
    if bundle is None or run is None or (bundle.run_id, bundle.revision) != (run.run_id, run.revision):
        return None
    keys = {instance: canonical for instance, canonical, _ in bundle.inventory_keys}
    cards = bundle.state.deck.all_cards
    if (not keys or len(keys) != len(bundle.inventory_keys)
            or len(cards) != len(keys) or {ref.instance_id for ref in cards} != set(keys)):
        return None
    for ref in cards:
        definition = bundle.kernel.cards.get(ref.card_id)
        if definition is None or (definition.canonical_card_id or ref.card_id) != keys[ref.instance_id]:
            return None
    return dict(run_id=run.run_id, revision=run.revision, canonical_card_ids=sorted(set(keys.values())),
        scope='navigation_inventory_family_only')


def retryable_low_confidence_family(prediction, family_ids, minimum_margin):
    """Permit discarding one weak batch, never accepting its identity for input."""
    return (prediction.accepted is False
        and prediction.recognizer == 'card_art_embedding_top5_hif_masked_v1'
        and prediction.reason == 'below_acceptance_threshold'
        and math.isfinite(prediction.confidence) and 0 <= prediction.confidence <= 1
        and math.isfinite(prediction.margin) and max(.05, minimum_margin) <= prediction.margin <= 1
        and bool(prediction.top_k_card_ids)
        and tuple(prediction.top_k_card_ids[0]) == tuple(family_ids))


def in_candidate_band(prediction, family_ids, minimum_margin):
    scores = prediction.top_k_scores
    groups = prediction.top_k_card_ids
    complete_three_card_art = (
        prediction.recognizer == 'card_art_embedding_top5_live_partial_v1'
        and prediction.reason == 'below_margin_threshold'
        and prediction.view_mode == 'full' and prediction.visible_ratio == 1.0
        and tuple(prediction.box[2:]) == (190, 250)
        and groups and set(groups[0]) == set(family_ids))
    if (prediction.accepted or not (prediction.recognizer == 'card_art_embedding_top5_hif_masked_v1'
                                   or complete_three_card_art)
            or prediction.reason not in ('below_acceptance_threshold', 'below_margin_threshold')
            or not math.isfinite(prediction.confidence) or not 0 <= prediction.confidence <= 1
            or not math.isfinite(prediction.margin) or not 0 <= prediction.margin < max(.05, minimum_margin)
            or not 1 <= len(scores) == len(groups) <= 5
            or any(not math.isfinite(value) for value in scores)
            or any(a < b for a, b in zip(scores, scores[1:]))):
        return False
    matching = [i for i, group in enumerate(groups) if set(family_ids) <= set(group)
        and 0 <= scores[0] - scores[i] <= max(.05, minimum_margin)]
    return len(matching) == 1 and set(groups[matching[0]]) == set(family_ids)


def _same_prediction(prediction, proof):
    if (not isinstance(proof, dict) or proof.get('scope') != SCOPE
            or proof.get('upgrade_verified') is not False
            or proof.get('slot') != prediction.slot or proof.get('frame_id') != prediction.frame_id
            or not .95 <= proof.get('confidence', 0) <= 1):
        return False
    raw = proof.get('original_prediction', {})
    current = prediction.to_dict()
    # Resolution later changes card_id, accepted, reason and effective confidence.
    # It must not change the observed geometry, frame or original model evidence.
    return all(raw.get(key) == current[key] for key in (
        'slot', 'box', 'frame_id', 'margin', 'raw_score', 'model_sha256',
        'classes_sha256', 'gallery_sha256', 'top_k_card_ids', 'top_k_scores'))


def family(prediction, proof=None):
    from .hif_inspected_art import inspected_family

    if _same_prediction(prediction, proof):
        return tuple(proof['family'])
    inspected = inspected_family(prediction, proof)
    if inspected:
        return inspected
    return tuple(prediction.top_k_card_ids[0]) if prediction.top_k_card_ids else ()


def valid_proof(context, prediction, proof):
    current = binding(context, prediction.slot)
    return (current is not None and _same_prediction(prediction, proof)
        and all(proof.get(key) == value for key, value in current.items())
        and current['canonical_card_id'] in proof.get('family', ()))


def bind_complete_base_title(context, prediction, proof):
    """Bind the existing high-margin complete-title proof to its verified hand.

    Historical callers must first verify the original selection and full A
    generation. This supplies source identity only, never fresh input evidence.
    """
    from .upgrade import normalize_upgrade_title

    current = binding(context, prediction.slot)
    if (current is None or not isinstance(proof, dict) or proof.get('scope') is not None
            or proof.get('accepted') is not True
            or proof.get('reason') != 'complete_base_title_and_visual_family_agree'
            or any(proof.get(key) != current[key] for key in ('run_id','revision','card_instance_id'))
            or not _same_prediction(prediction, dict(proof,scope=SCOPE))):
        return None
    raw = proof['original_prediction']
    family_ids = tuple(proof.get('family', ()))
    original = SimpleNamespace(**raw)
    if (not retryable_low_confidence_family(original,family_ids,.05)
            or current['canonical_card_id'] not in family_ids
            or prediction.reason != 'bound_complete_title_and_visual_family'
            or proof['confidence'] != prediction.confidence):
        return None
    bundle = context.gkh_hif_opening_state
    definition = bundle.kernel.cards[bundle.state.deck.hand[prediction.slot].card_id]
    normalize = lambda value: normalize_upgrade_title(''.join(value.split())).rstrip('+')
    box, roi = tuple(prediction.box), tuple(proof.get('roi', ()))
    if (not isinstance(proof.get('text'),str) or normalize(proof['text']) != normalize(definition.name)
            or len(roi) != 4 or any(type(value) is not int for value in roi)
            or roi[0] != box[0]+2 or roi[1] < box[1]+box[3]-36
            or roi[1]+roi[3] > box[1]+box[3] or not 0 < roi[2] <= box[2]-2
            or not 0 < roi[3] <= 36 or roi[0]+roi[2] > box[0]+box[2]):
        return None
    return dict(proof,**current,scope=SCOPE,visual_candidate_rank=0,visual_score_gap=0.,
                original_top_k_card_ids=raw['top_k_card_ids'],original_top_k_scores=raw['top_k_scores'])


def event_proof(event, prediction):
    from .hif_inspected_art import inspected_family

    matches = [proof for proof in event.get('bound_complete_title_families', ())
        if _same_prediction(prediction, proof)]
    matches.extend(proof for proof in event.get('inspected_title_art_families', ())
        if inspected_family(prediction, proof))
    return matches[0] if len(matches) == 1 else None


def selected_detail_family_valid(detail, *, frame_ids, family_ids, card_id, catalog, plan):
    """Require fresh complete titles to identify one catalog family globally."""
    from .upgrade import normalize_upgrade_title

    normalize = lambda value: normalize_upgrade_title(''.join(value.split()))
    try:
        name = normalize(catalog.entries[card_id].name)
        matching = {key for key, entry in catalog.entries.items()
                    if entry.plan in ('free',plan) and normalize(entry.name) == name}
        family = {key for key, entry in catalog.entries.items()
                  if entry.plan in ('free',plan) and normalize(entry.name).rstrip('+') == name.rstrip('+')}
        if (not name or matching != {card_id} or family != set(family_ids)
                or detail.get('accepted') is not True or detail.get('card_id') != card_id
                or len(frame_ids) != 3 or len(set(frame_ids)) != 3
                or len(detail.get('observations', ())) != 3):
            return False
        for fid, row in zip(frame_ids, detail['observations'], strict=True):
            roi, texts = row['roi'], row['ocr']
            if (row.get('frame_id') != fid or row.get('card_id') != card_id or len(texts) != 1
                    or len(roi) != 4 or any(type(x) is not int for x in roi)
                    or roi[0] < 0 or roi[1] < 360 or min(roi[2:]) <= 0
                    or roi[0]+roi[2] > 720 or roi[1]+roi[3] > 1280):
                return False
            text = texts[0]
            box = text['box']
            if (type(text.get('score')) not in (int,float) or not .95 <= text['score'] <= 1
                    or normalize(text['text']) != name or len(box) != 4
                    or any(type(x) is not int for x in box) or min(box[:2]) < 0
                    or min(box[2:]) <= 0 or box[0]+box[2] > roi[2] or box[1]+box[3] > roi[3]):
                return False
        return True
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def selected_singleton_title_evidence(context, frames, candidates, prediction, detail, *,
                                      family_ids, card_id, catalog, plan):
    """Corroborate an occluded singleton using its two complete title surfaces.

    The large score preview can remove the family from both artwork rankings.
    An already bound singleton still has a full card title and a separate
    detail title. Require both on every frame; weak art remains diagnostic.
    """
    import cv2
    import numpy as np

    from .upgrade import normalize_upgrade_title
    from .inventory import _inventory_ocr
    from .hif_visual import hif_selected_row_proposal
    try:
        from agent.card_selection.model import frame_identifier
    except ModuleNotFoundError:
        from card_selection.model import frame_identifier

    frame_ids = tuple(frame_identifier(frame) for frame in frames)
    if (len(candidates) != 1 or prediction.slot != 0 or prediction.accepted
            or prediction.recognizer != 'card_art_embedding_top5_live_partial_v1'
            or prediction.reason not in ('below_acceptance_threshold', 'below_margin_threshold')
            or prediction.view_mode != 'full' or prediction.visible_ratio != 1
            or tuple(prediction.box[2:]) != (190, 250)
            or tuple(candidates[0].box) != tuple(prediction.box)
            or prediction.frame_id not in frame_ids
            or not selected_detail_family_valid(detail, frame_ids=frame_ids,
                family_ids=family_ids, card_id=card_id, catalog=catalog, plan=plan)):
        return None
    normalize = lambda value: normalize_upgrade_title(''.join(value.split()))
    expected = normalize(catalog.entries[card_id].name)
    observations = []
    for frame, frame_id in zip(frames, frame_ids, strict=True):
        proposal = hif_selected_row_proposal(frame, candidates, hand_count=1)
        if (proposal is None or proposal[1] != 0
                or tuple(proposal[0][0].box) != tuple(prediction.box)):
            return None
        x, y, w, h = prediction.box
        top = y+h-36
        strip = frame[top:y+h, x:x+w, :3].astype(np.int16)
        white = ((strip.min(axis=2) >= 210)
            & (strip.max(axis=2)-strip.min(axis=2) <= 24)).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(white)
        plates = [(x+int(px), top+int(py), int(pw), int(ph))
            for px, py, pw, ph, area in stats[1:]
            if 180 <= pw <= 190 and 30 <= ph <= 34 and area/(pw*ph) >= .65]
        if len(plates) != 1:
            return None
        px, py, pw, ph = plates[0]
        rows = _inventory_ocr(context, frame[py:py+ph, px:px+pw].copy(),
            'HifSelectedSingletonCompleteTitle', only_rec=True)
        if len(rows) != 1 or not .95 <= rows[0].score <= 1 or normalize(rows[0].text) != expected:
            return None
        observations.append(dict(frame_id=frame_id, roi=list(plates[0]),
            text=rows[0].text, confidence=rows[0].score))
    return dict(scope='selected_singleton_two_complete_titles', observations=observations,
        detail=detail, card_id=card_id, upgrade_verified=False, submission_authorized=False)


def margin_evidence(context, certificate, fresh, event, minimum_margin):
    source = certificate['source_prediction']
    probe = certificate.get('bound_title_evidence')
    current = event_proof(event, fresh)
    from .hif_inspected_art import source_title_evidence_valid

    inspected_source = source_title_evidence_valid(context, certificate)
    probe_ok = (source.margin >= minimum_margin or valid_proof(context, source, probe)
                or inspected_source and current is not None
                and current.get('source_identity_kind') == 'inspected_art')
    fresh_ok = (fresh.accepted and (fresh.margin >= minimum_margin
        or valid_proof(context, fresh, current)))
    if (not probe_ok or not fresh_ok or family(source, probe) != tuple(certificate['family'])
            or family(fresh, current) != tuple(certificate['family'])):
        return None
    return dict(slot=fresh.slot, probe_margin=source.margin, fresh_margin=fresh.margin,
        probe_uses_bound_title=source.margin < minimum_margin,
        fresh_uses_bound_title=fresh.margin < minimum_margin, upgrade_verified=False,
        **({'probe_uses_inspected_art': True} if inspected_source else {}))


def historical_title_evidence(bundle, complete_event, refresh, source, family_ids, minimum_margin):
    """Recheck a saved A title as historical identity, never as fresh evidence."""
    from .opening_observer import normalize_upgrade_title

    opening = complete_event.get('event') == 'live_opening_state'
    context = SimpleNamespace(gkh_hif_opening_state=bundle,
        gkh_hif_live_context=SimpleNamespace(run_id=bundle.run_id, revision=bundle.revision),
        gkh_hif_committed_event=None if opening else complete_event,
        gkh_hif_complete_opening_event=complete_event if opening else None)
    slot = source.get('slot')
    if type(slot) is not int:
        return None
    current = binding(context, slot)
    matches = [row for row in refresh.get('bound_complete_title_families', ())
               if row.get('slot') == slot]
    if current is None or len(matches) != 1:
        return None
    saved = matches[0]
    proof = dict(saved)
    # JSON changes only these nested sequences; never import a saved proof as
    # a fresh-frame credential. The original model evidence stays untouched.
    proof['hand_instance_ids'] = tuple(saved.get('hand_instance_ids', ()))
    key = saved.get('inventory_key', ())
    if len(key) != 3:
        return None
    proof['inventory_key'] = (key[0], key[1], tuple(tuple(x) for x in key[2]))
    observed = SimpleNamespace(**source, to_dict=lambda: source)
    raw = saved.get('original_prediction', {})
    full_art = saved.get('complete_full_art')
    raw_prediction = SimpleNamespace(**raw)
    visual_valid = (full_view_art_evidence_valid(full_art, raw_prediction, family_ids, minimum_margin)
        if full_art is not None else bool(raw) and (
            in_candidate_band(raw_prediction, family_ids, minimum_margin)
            or retryable_low_confidence_family(raw_prediction, family_ids, minimum_margin)))
    if saved.get('accepted_visual_family_corroborated') is True:
        visual_valid = bool(raw and raw.get('accepted') is True
            and raw.get('reason') == 'accepted_visual_family'
            and raw.get('recognizer') in ('card_art_embedding_top5_live_partial_v1',
                'card_art_embedding_top5_hif_masked_v1', 'card_art_embedding_full_masked_agreement_v1')
            and raw.get('view_mode') == 'full' and raw.get('visible_ratio') == 1
            and tuple(raw.get('box', ())[2:]) == (190, 250)
            and raw.get('top_k_card_ids') and tuple(raw['top_k_card_ids'][0]) == tuple(family_ids)
            and raw.get('margin', -1) >= max(.05, minimum_margin))
    if (not valid_proof(context, observed, proof)
            or source.get('frame_id') not in refresh.get('frame_ids', ())
            or tuple(saved.get('family', ())) != tuple(family_ids)
            or source.get('reason') != 'bound_complete_title_and_visual_family'
            or saved.get('accepted') is not True
            or saved.get('reason') != 'complete_base_title_and_visual_family_agree'
            or saved.get('confidence') != source.get('confidence')
            or not visual_valid
            or saved.get('original_top_k_card_ids') != raw.get('top_k_card_ids')
            or saved.get('original_top_k_scores') != raw.get('top_k_scores')):
        return None
    if full_art is None:
        ranks = [i for i, ids in enumerate(raw['top_k_card_ids']) if tuple(ids) == tuple(family_ids)]
        if (len(ranks) != 1 or saved.get('visual_candidate_rank') != ranks[0]
                or saved.get('visual_score_gap') != raw['top_k_scores'][0]-raw['top_k_scores'][ranks[0]]):
            return None
    elif saved.get('visual_candidate_rank') is not None or saved.get('visual_score_gap') is not None:
        return None
    definition = bundle.kernel.cards[bundle.state.deck.hand[slot].card_id]
    normalize = lambda text: normalize_upgrade_title(''.join(text.split())).rstrip('+')
    if not isinstance(saved.get('text'), str) or normalize(saved['text']) != normalize(definition.name):
        return None
    roi, box = saved.get('roi', ()), source.get('box', ())
    if (len(roi) != 4 or len(box) != 4 or any(type(x) is not int for x in (*roi, *box))):
        return None
    x, y, w, h = box
    px, py, pw, ph = roi
    if not (px == x+2 and py >= y+h-36 and py+ph <= y+h
            and 0 < pw <= w-2 and 0 < ph <= 36 and px+pw <= x+w):
        return None
    return saved


def selected_neighbor_margin_evidence(certificate, fresh, revalidation):
    """Consume two in-process carry results; neither claims a new title read."""
    probe = certificate.get('selected_neighbor_identity_carry') or {}
    current = revalidation or {}
    if (probe.get('acceptance_mode') != 'selected_right_neighbor_bound_title_certificate'
            or current.get('acceptance_mode') != probe.get('acceptance_mode')
            or not probe.get('source_title_evidence')
            or probe.get('source_title_evidence') != current.get('source_title_evidence')
            or probe.get('source_prediction') != current.get('source_prediction')
            or any(row.get('accepted') is not True or row.get('carried') is not True
                   or row.get('standard_accepted') is not False for row in (probe, current))):
        return None
    for prediction, carry in ((certificate['source_prediction'], probe), (fresh, current)):
        raw = carry.get('carried_prediction', {})
        actual = prediction.to_dict()
        # Card resolution may change IDs/class/reason only. Every observed
        # pixel identity, original ranking and confidence must still agree.
        if any(raw.get(key) != actual.get(key) for key in (
                'slot', 'box', 'frame_id', 'confidence', 'margin', 'raw_score',
                'model_sha256', 'classes_sha256', 'gallery_sha256',
                'top_k_card_ids', 'top_k_scores')):
            return None
        if tuple((raw.get('top_k_card_ids') or [()])[0]) != tuple(certificate['family']):
            return None
    return dict(slot=fresh.slot, probe_margin=certificate['probe_margin'],
        fresh_margin=fresh.margin, source_frame_id=current['source_prediction']['frame_id'],
        fresh_frame_id=current['fresh_prediction']['frame_id'],
        reason='historical_complete_title_and_fresh_selected_overlap', upgrade_verified=False)
