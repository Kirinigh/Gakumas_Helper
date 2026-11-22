"""Narrow, observation-only permission to preview one bound gray hand card.

The host owns the existing Maa endpoint and 1000 ms dispatch checks. This
permission never makes a detector candidate clickable and never permits a
business submit. It must be journaled before dispatch; any prior journal blocks
another selection attempt until the native transaction is explicitly recovered.
"""

import copy
import json
import math
from types import SimpleNamespace
from dataclasses import asdict, replace

from .contracts import LiveAction, DecisionProposal
from .high_score.kernel import EffectKind, CardCostKind
from .high_score.contracts import PlanType, ActionKind, DecisionKind


def _require(condition, reason):
    if not condition:
        raise ValueError('gray singleton preview: ' + reason)


def _three_frames(values):
    return (isinstance(values, (tuple, list)) and len(values) == 3
            and all(isinstance(value, str) and value for value in values)
            and len(set(values)) == 3)


def verify_recorded_gray_selection_reservation(bundle, journal, *, slot, box, generation):
    """Recheck the native loader's saved reservation, never authorize a submit."""
    permit = journal.get('permit')
    fresh_a = journal.get('refresh_a', {})
    ref = bundle.state.deck.hand[slot]
    card = bundle.kernel.cards[ref.card_id]
    card_id = card.canonical_card_id or card.card_id
    if (not isinstance(permit, dict) or permit.get('accepted') is not True
            or permit.get('phase') != 'selection_reserved'
            or permit.get('run_id') != bundle.run_id or permit.get('revision') != bundle.revision
            or permit.get('card_instance_id') != ref.instance_id or permit.get('card_id') != card_id
            or permit.get('slot') != slot or permit.get('box') != box
            or permit.get('maximum_selection_count') != 1 or permit.get('maximum_submit_count') != 0
            or permit.get('submission_authorized') is not False
            or permit.get('recovery_required_after_dispatch') is not True
            or permit.get('frame_ids') != fresh_a.get('frame_ids')
            or json.dumps(permit.get('generation'), sort_keys=True) != json.dumps(generation, sort_keys=True)
            or fresh_a.get('accepted') is not False
            or fresh_a.get('reason') != 'provider_stop:hif_policy_card_not_playable'):
        raise ValueError('native gray preview reservation is missing or inconsistent')


def carry_selected_right_neighbor_identity(bundle, *, complete_event, journal,
        prediction, evidence, identity_threshold, identity_min_margin, geometry_comparator):
    """Carry the original unselected A identity behind one selected right overlap.

    The fresh low margin remains diagnostic evidence. This is not a new visual
    acceptance or an input permit; selected recovery still requires fresh A/B.
    """
    fresh = prediction.to_dict() if hasattr(prediction, 'to_dict') else copy.deepcopy(prediction)
    a = journal.get('refresh_a', {})
    action = a.get('proposal', {}).get('candidate_traces', {}).get('high_score', {}).get('action', {})
    selected, slot = action.get('source_slot'), fresh.get('slot')
    hand = bundle.state.deck.hand
    _require(4 <= len(hand) <= 5 and type(selected) is int and 0 <= selected < len(hand)-1
             and type(slot) is int and slot == selected+1
             and action.get('kind') == 'PLAY_CARD'
             and action.get('card_instance_id') == hand[selected].instance_id,
             'selected overlap is not the bound immediate right neighbor')
    _require(journal.get('ordinary_safe_preview') is True and journal.get('selection_ready') is True
             and journal.get('selection_count') == 1 and journal.get('business_submit_count') == 0,
             'selected overlap lacks one unsubmitted selection')
    dispatches = [item for item in journal.get('selection_events', ())
                  if item.get('phase') == 'execution_selection']
    _require(len(dispatches) == 1 and dispatches[0].get('accepted') is True
             and dispatches[0].get('click_dispatched') is True
             and dispatches[0].get('attempted_slot') == selected,
             'selected overlap lacks its actual selection dispatch')
    _require(complete_event.get('accepted') is True
             and complete_event.get('run_id') == bundle.run_id
             and complete_event.get('revision') == bundle.revision
             and complete_event.get('checks') and all(v is True for v in complete_event['checks'].values()),
             'selected overlap complete binding differs')
    generation = dict(turn=bundle.state.turn, turns_remaining=bundle.state.turns_remaining,
        stamina=bundle.state.stamina, shield=bundle.state.genki, score=bundle.state.score,
        score_multiplier=bundle.state.score_multiplier, hand_count=len(hand), plan='sense',
        action_order=list(complete_event['core']['fields'][5]))
    _require(a.get('accepted') is True and a.get('generation') == generation
             and a.get('expected_selected_slot') is None and a.get('unselected_geometry_authorized') is True
             and a.get('select_consensus', {}).get('observations') == [[], [], []],
             'selected overlap original A is not the bound unselected state')
    identities, certificates = a.get('identity_revalidation', ()), journal.get('identity_certificate', ())
    _require(len(identities) == len(certificates) == len(hand), 'selected overlap source hand is incomplete')
    source_row, certificate = identities[slot], certificates[slot]
    source = source_row.get('fresh_prediction') or {}
    family = list(certificate.get('family', ()))
    ref = hand[slot]
    canonical = bundle.kernel.cards[ref.card_id].canonical_card_id or ref.card_id
    from .hif_bound_title import in_candidate_band, historical_title_evidence

    title_evidence = historical_title_evidence(
        bundle, complete_event, a, source, family, identity_min_margin)
    # A proven half-dark layer can leave the covered neighbour's narrow art
    # ambiguous. Keep that real ranking, but allow the original strong A to
    # resolve its existing candidate band within this one unsubmitted action.
    # A title-assisted A and an independently strong conflicting B cannot use it.
    normalized_band = False
    if (title_evidence is None and list((source.get('top_k_card_ids') or [()])[0]) == family
            and type(source.get('margin')) in (int, float)
            and source['margin'] >= identity_min_margin):
        pixel_proofs = [p for p in evidence.get('art_preprocessing_evidence', ())
                        if p.get('slot') == slot and p.get('frame_id') == fresh.get('frame_id')]
        if len(pixel_proofs) == 1:
            pixels = pixel_proofs[0]
            normalized = pixels.get('normalized_prediction', {})
            scores, groups = fresh.get('top_k_scores', ()), fresh.get('top_k_card_ids', ())
            matching = [i for i, group in enumerate(groups) if list(group) == family]
            normalized_band = bool(
                pixels.get('profile') == 'observed_disabled_half_intensity_v1'
                and pixels.get('selected_slot') == selected and pixels.get('gain') == 2
                and type(pixels.get('gray_level')) is int and type(pixels.get('white_level')) is int
                and pixels['gray_level']*2 == pixels['white_level'] >= 235
                and pixels.get('original_prediction', {}).get('accepted') is False
                and normalized.get('accepted') is False
                and fresh.get('reason') == normalized.get('reason') == 'below_margin_threshold'
                and all(normalized.get(key) == fresh.get(key) for key in
                        ('slot', 'frame_id', 'box', 'confidence', 'margin', 'top_k_card_ids', 'top_k_scores'))
                and family in [list(group) for group in pixels.get('source_families', ())]
                and len(matching) == 1 and len(scores) == len(groups)
                and type(scores[matching[0]]) in (int, float)
                and scores[matching[0]] >= identity_threshold
                and in_candidate_band(SimpleNamespace(**fresh), family, identity_min_margin))
    _require(certificate.get('slot') == slot and certificate.get('card_id') == canonical
             and certificate.get('upgraded') is ref.upgraded and canonical in family
             and source_row.get('slot') == slot and source_row.get('accepted') is True
             and source_row.get('standard_accepted') is True and source_row.get('carried') is False
             and source_row.get('acceptance_mode') == 'fresh_exact_family'
             and source_row.get('fresh_family') == source_row.get('expected_family') == family
             and source.get('accepted') is True and source.get('slot') == slot
             and (list((source.get('top_k_card_ids') or [()])[0]) == family or title_evidence is not None)
             and (list((fresh.get('top_k_card_ids') or [()])[0]) == family or normalized_band),
             'selected overlap original or current family differs')
    for name, value in (('source confidence', source.get('confidence')),
                        ('source margin', source.get('margin')),
                        ('fresh confidence', fresh.get('confidence')),
                        ('fresh margin', fresh.get('margin'))):
        _require(type(value) in (int, float) and math.isfinite(value), 'selected overlap invalid '+name)
    _require(source['confidence'] >= identity_threshold
             and (source['margin'] >= identity_min_margin or title_evidence is not None)
             and fresh['confidence'] >= identity_threshold and 0 <= fresh['margin'] < identity_min_margin
             and fresh.get('accepted') is False and fresh.get('view_mode') == 'left_occluded'
             and fresh.get('visible') is True and fresh.get('recognizer') == source.get('recognizer')
             and all(isinstance(source.get(key), str) and source[key] and source[key] == fresh.get(key)
                     for key in ('model_sha256', 'classes_sha256', 'gallery_sha256')),
             'selected overlap lacks high-margin original evidence or matching masked recognition')
    frames = evidence.get('frame_ids', ())
    consensus, selections = evidence.get('select_consensus') or {}, evidence.get('select_evidence', ())
    _require(_three_frames(frames) and not set(frames) & set(a.get('frame_ids', ()))
             and fresh.get('frame_id') in frames and source.get('frame_id') in a.get('frame_ids', ())
             and evidence.get('geometry_stable') is True and consensus.get('accepted') is True
             and consensus.get('selected_slot') == selected
             and list(map(list, consensus.get('observations', ()))) == [[selected]]*3
             and len(selections) == 3 and all(item.get('geometry_available') is True
                 and item.get('geometry_ambiguous') is False and item.get('ocr_slots') == [selected]
                 and item.get('geometry_raised_slots') == [selected] and item.get('resolved_slots') == [selected]
                 for item in selections), 'selected overlap lacks fresh unique SELECT and raised geometry')
    boxes, old_boxes = evidence.get('candidate_boxes', ()), a.get('candidate_boxes', ())
    _require(len(boxes) == len(old_boxes) == 3 and callable(geometry_comparator)
             and all(len(row) == len(hand) and all(len(box) == 4 for box in row) for row in (*boxes, *old_boxes)),
             'selected overlap geometry is incomplete')
    reference = old_boxes[-1]
    _require(list(fresh.get('box', ())) == list(boxes[list(frames).index(fresh['frame_id'])][slot]),
             'selected overlap prediction frame geometry differs')
    for row in boxes:
        left, right = row[selected], row[slot]
        _require(0 < left[0]+left[2]-right[0] < right[2]
                 and 16 <= reference[selected][1]-left[1] <= 36
                 and right[1] > left[1], 'selected card does not physically cover its right neighbor')
        lowered = [list(box) for box in row]
        lowered[selected][1] = reference[selected][1]
        _require(geometry_comparator(tuple(SimpleNamespace(box=box) for box in reference),
                                    tuple(SimpleNamespace(box=box) for box in lowered)),
                 'selected overlap physical hand differs from the original A')
    carried = dict(source, box=copy.deepcopy(fresh['box']), accepted=True,
        reason='live_selected_right_neighbor_identity_carried',
        **{key: fresh.get(key) for key in ('detector_label', 'detector_score', 'visible', 'clickable')})
    if title_evidence is not None:
        # Unlike a high-margin original, A's raw top-one family was ambiguous.
        # Keep the current matching top-one and its real low margin, with the
        # historical title stored separately as the continuity justification.
        carried = dict(fresh, accepted=True, reason='live_selected_right_neighbor_identity_carried')
    return dict(accepted=True, carried=True, standard_accepted=False,
        acceptance_mode=('selected_right_neighbor_bound_title_certificate' if title_evidence is not None
                         else 'selected_right_neighbor_unselected_certificate'),
        slot=slot, selected_slot=selected, card_id=canonical, upgraded=ref.upgraded,
        run_id=bundle.run_id, revision=bundle.revision, submission_authorized=False,
        source_prediction=copy.deepcopy(source), fresh_prediction=copy.deepcopy(fresh),
        carried_prediction=carried, frame_ids=list(frames),
        **(dict(normalized_candidate_band=True,
                art_preprocessing_evidence=copy.deepcopy(pixel_proofs[0])) if normalized_band else {}),
        **(dict(source_title_evidence=copy.deepcopy(title_evidence)) if title_evidence is not None else {}))


def _complete_gray_hand_boxes(boxes, *, exact=True):
    """Check already observed footprints against the existing HIF row geometry."""
    if not (1 <= len(boxes) <= 5 and all(
        len(box) == 4 and all(type(value) is int for value in box)
        and box[2] > 0 and box[3] > 0
        and (not exact and len(boxes) <= 3 or tuple(box[2:]) == (190, 250))
        and 0 <= box[0] <= 720-box[2] and 0 <= box[1] <= 1280-box[3]
        for box in boxes)):
        return False
    if len(boxes) <= 3:
        return all(left[0]+left[2] <= right[0] for left, right in zip(boxes, boxes[1:]))
    from .selection import live_selection_geometry_evidence
    from .hif_visual import hif_select_label_rois

    candidates = tuple(SimpleNamespace(slot=slot, box=tuple(box)) for slot, box in enumerate(boxes))
    try:
        hif_select_label_rois((1280, 720), candidates)
    except ValueError:
        return False
    geometry = live_selection_geometry_evidence(candidates)
    # These checks corroborate the complete pixel-backed row already supplied
    # by A/B; spacing alone never supplies a missing card or its identity.
    return (geometry['available'] and not geometry['ambiguous']
        and all(not (other[0] <= box[0]+box[2]//2 < other[0]+other[2]
                     and other[1] <= box[1]+box[3]//2 < other[1]+other[3])
                for slot, box in enumerate(boxes) for index, other in enumerate(boxes)
                if index != slot))


def _overlapping_gray_selection(boxes, slot):
    """Recompute selection geometry for the newly supported overlapping row."""
    if len(boxes) <= 3:
        return True
    from .selection import live_selection_geometry_evidence

    geometry = live_selection_geometry_evidence(tuple(
        SimpleNamespace(slot=index, box=tuple(box)) for index, box in enumerate(boxes)))
    return (geometry['available'] and not geometry['ambiguous']
            and geometry['selected_slot'] == slot
            and geometry['raised_slots'] == (() if slot is None else (slot,)))


def _gray_hand_identity(bundle, certificates, identities):
    hand = bundle.state.deck.hand
    return (len(certificates) == len(identities) == len(hand) and all(
        cert.get('slot') == item.get('slot') == slot
        and cert.get('card_id') == (bundle.kernel.cards[ref.card_id].canonical_card_id or ref.card_id)
        and cert.get('upgraded') is ref.upgraded and cert['card_id'] in cert.get('family', ())
        and item.get('accepted') is True and item.get('standard_accepted') is True
        and item.get('carried') is False
        and tuple(item.get('fresh_family', ())) == tuple(cert.get('family', ()))
        for slot, (ref, cert, item) in enumerate(zip(hand, certificates, identities, strict=True))))


def _gray_source_binding(bundle, event):
    if event.get('verification_scope') in ('rule_maintained_information_boundary_v1', 'rule_maintained_preinput_v1'):
        _ordinary_selected_rule_binding(bundle, event)
        return
    _require(event.get('accepted') is True
             and event.get('run_id') == bundle.run_id and event.get('revision') == bundle.revision
             and isinstance(event.get('checks'), dict)
             and {'core', 'hand_identity', 'all_card_zones', 'drinks', 'turns'} <= set(event['checks'])
             and all(value is True for value in event['checks'].values()),
             'complete native binding is not verified')


def authorize_gray_singleton_preview(bundle, live, *, complete_event, candidates,
        identity_certificate, refresh_event, planned_action, safety, prior_journal=None):
    """Return a durable one-selection reservation, never PLAY authorization.

    The target is a kernel-legal NORMAL card with zero catalog cost and no
    direct effects. Every physical footprint must be complete and independently
    bound by fresh A, including the existing four/five-card overlapping row.
    The historical function name remains for callers.
    """
    _require(prior_journal is None, 'an earlier selection reservation needs recovery')
    live.validate()
    _require(not live.unknown_fields and not live.contradictory_fields, 'incomplete live state')
    count = len(bundle.state.deck.hand)
    _require(1 <= count <= 5 and count == len(live.hand) == len(candidates)
             == len(identity_certificate), 'hand must be complete and physically bound')
    slot = getattr(planned_action, 'source_slot', None)
    _require(type(slot) is int and 0 <= slot < count, 'policy target slot is invalid')
    ref, observed, candidate, certificate = (
        bundle.state.deck.hand[slot], live.hand[slot], candidates[slot], identity_certificate[slot])
    definition = bundle.kernel.cards[ref.card_id]
    card_id = definition.canonical_card_id or definition.card_id
    _require(definition.cost_kind is CardCostKind.NORMAL and definition.stamina_cost == 0
             and not definition.effects, 'unsupported catalog cost or direct effects')
    _require(planned_action is not None and planned_action.kind is ActionKind.PLAY_CARD
             and planned_action.card_instance_id == ref.instance_id
             and planned_action.source_slot == slot
             and planned_action.drink_id is None and planned_action.target_id is None,
             'policy target differs from the bound card')
    legal = bundle.kernel.legal_actions(bundle.state, plan_type=PlanType.SENSE,
                                       decision_kind=DecisionKind.LIVE)
    _require(any(action.kind is ActionKind.PLAY_CARD and action.card_instance_id == ref.instance_id
                 for action in legal), 'kernel rejects this play')
    _gray_source_binding(bundle, complete_event)
    core = complete_event.get('core', {})
    _require(_three_frames(core.get('frame_ids')), 'complete binding lacks source frames')
    _require(tuple(map(tuple, core.get('families', ())))
             == tuple(tuple(cert.get('family', ())) for cert in identity_certificate),
             'complete hand family differs')
    expected_core = [live.turns_remaining, live.stamina, bundle.state.genki,
                     live.score, live.score_multiplier, list(live.action_order)]
    actual_core = list(core.get('fields', ()))
    if len(actual_core) == 6:
        actual_core[5] = list(actual_core[5])
    _require(actual_core == expected_core, 'complete core differs from fresh A')
    for name in ('turn', 'turns_remaining', 'stamina', 'max_stamina', 'score', 'score_multiplier'):
        _require(getattr(bundle.state, name) == getattr(live, name), 'bound scalar changed: ' + name)
    _require(all(card.slot == item.slot == cert.get('slot') == index
                 and card.card_id == cert.get('card_id') and card.identity_accepted
                 and card.upgraded is bound.upgraded and card.box == item.box
                 and item.visible is True
                 for index, (card, item, cert, bound) in enumerate(zip(
                     live.hand, candidates, identity_certificate, bundle.state.deck.hand, strict=True))),
             'complete physical hand identity differs')
    _require(observed.slot == candidate.slot == certificate.get('slot') == slot
             and observed.card_id == certificate.get('card_id') == card_id
             and observed.identity_accepted and observed.upgraded is ref.upgraded
             and certificate.get('upgraded') is ref.upgraded
             and card_id in certificate.get('family', ())
             and observed.cost == 0 and observed.playable is False
             and observed.box == candidate.box and candidate.visible is True
             and candidate.clickable is False
             and candidate.detector_label == 'hif_gray_observation',
             'gray identity, slot, or physical footprint is not exact')
    reference = [item.box for item in candidates]
    _require(_complete_gray_hand_boxes(reference) and _overlapping_gray_selection(reference, None),
             'physical footprint is outside normalized Live')
    event = refresh_event
    _require(event.get('phase') == 'authoritative_refresh_a'
             and event.get('expected_selected_slot') is None
             and event.get('unselected_geometry_authorized') is True
             and event.get('click_dispatched') is False
             and event.get('input_may_have_been_dispatched') is False
             and event.get('accepted') is False
             and event.get('reason') == 'provider_stop:hif_policy_card_not_playable',
             'A must reject only the gray playable gate')
    _require(_three_frames(event.get('frame_ids')) and live.snapshot_id == event['frame_ids'][-1]
             and not set(event['frame_ids']) & set(core['frame_ids']), 'fresh A frames are not independent')
    maximum = safety.config.max_snapshot_age_ms
    _require(type(maximum) in (int, float) and math.isfinite(maximum) and 0 < maximum <= 1000,
             'input evidence limit must remain at most 1000 ms')
    for name in ('capture_age_ms', 'evidence_window_age_ms'):
        value = event.get(name)
        _require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= maximum,
                 'stale or invalid A ' + name)
    _require(event['capture_age_ms'] == live.capture_age_ms, 'A snapshot age differs')
    consensus = event.get('select_consensus', {})
    _require(consensus.get('accepted') is True and consensus.get('selected_slot') is None
             and len(consensus.get('observations', ())) == 3
             and all(not slots for slots in consensus['observations']), 'A is not unanimously unselected')
    evidence = event.get('select_evidence', ())
    _require(len(evidence) == 3 and all(
        item.get('geometry_available') is True and item.get('geometry_ambiguous') is False
        and item.get('geometry_raised_slots') == [] and item.get('resolved_slots') == []
        for item in evidence), 'A lacks independent unselected geometry')
    boxes = event.get('candidate_boxes', ())
    _require(len(boxes) == 3 and all(len(row) == count and _complete_gray_hand_boxes(row)
        and _overlapping_gray_selection(row, None)
        and all(abs(box[0]-old[0]) <= 8 and abs(box[1]-old[1]) <= 8
                for box, old in zip(row, reference, strict=True)) for row in boxes),
        'A full hand geometry is incomplete or unstable')
    identities = event.get('identity_revalidation', ())
    _require(_gray_hand_identity(bundle, identity_certificate, identities),
             'A lacks independent fresh identity')
    _require(event.get('health_votes') == event.get('playable_votes') == 3
             and event.get('secondary_votes') == event.get('popup_votes') == 0,
             'Live anchors or overlays are not clear')
    resource = event.get('resource_transaction', {})
    _require(resource.get('accepted') is True and resource.get('selected_slot') is None
             and resource.get('normalized_stamina') == live.stamina
             and resource.get('normalized_shield') == bundle.state.genki,
             'A resource transaction is not unchanged')
    # Check every existing safety rule with a hypothetical playable bit, but
    # return only a preview reservation. No ALLOW decision escapes this helper.
    preview = replace(live, hand=tuple(replace(card, playable=True) if index == slot else card
                                      for index, card in enumerate(live.hand)))
    proposal = DecisionProposal('PROPOSE', 'gray_singleton_preview_only', LiveAction('PLAY_CARD', slot),
                                1., 'hif-gray-preview', '1', 0.)
    _require(safety.filter(preview, proposal).has_action_intent, 'another existing safety gate rejects A')
    return dict(phase='selection_reserved', accepted=True, run_id=bundle.run_id,
        revision=bundle.revision, snapshot_id=live.snapshot_id, card_id=card_id,
        card_instance_id=ref.instance_id, slot=slot, box=list(candidate.box),
        frame_ids=list(event['frame_ids']), generation=copy.deepcopy(event.get('generation')),
        core=expected_core, maximum_selection_count=1, maximum_submit_count=0,
        submission_authorized=False, recovery_required_after_dispatch=True)


def _same_selected_geometry(boxes, reference, geometry_comparator):
    """Keep a complete final hand and the host's existing detector tolerance."""
    return (callable(geometry_comparator) and _complete_gray_hand_boxes(reference)
        and len(boxes) == 3 and all(len(row) == len(reference)
            and _complete_gray_hand_boxes(row, exact=False)
            and geometry_comparator(tuple(SimpleNamespace(box=tuple(box)) for box in reference),
                                    tuple(SimpleNamespace(box=tuple(box)) for box in row)) for row in boxes))


def _gray_selected_hand_movement(boxes, original, slot, geometry_comparator):
    if (len(original) != 3 or not _same_selected_geometry(original, original[-1], geometry_comparator)
            or not _same_selected_geometry(boxes, boxes[-1] if len(boxes) == 3 else (), geometry_comparator)
            or len(original[-1]) != len(boxes[-1]) or not 0 <= slot < len(original[-1])):
        return False
    reference = original[-1]
    if any(not _overlapping_gray_selection(row, None) for row in original):
        return False
    for row in boxes:
        if (not 16 <= reference[slot][1]-row[slot][1] <= 36
                or not _overlapping_gray_selection(row, slot)):
            return False
        lowered = [list(box) for box in row]
        lowered[slot][1] = reference[slot][1]
        if len(reference) >= 4 and any(abs(current[axis]-old[axis]) > 8
                for current, old in zip(lowered, reference, strict=True) for axis in (0, 1)):
            return False
        if not geometry_comparator(tuple(SimpleNamespace(box=tuple(box)) for box in reference),
                                   tuple(SimpleNamespace(box=tuple(box)) for box in lowered)):
            return False
    return True


def selected_zero_cost_baseline(bundle, *, complete_event, journal, evidence, geometry_comparator):
    """Recover resources after a recorded selection, without granting PLAY.

    The native loader proves journal provenance and same-process recovery.
    These newly observed frames must independently prove SELECT, the complete
    title/body and the zero-cost resource transaction. Input still requires
    the ordinary subsequent fresh A/B, policy, safety and endpoint checks.
    """
    hand = bundle.state.deck.hand
    permit = journal.get('permit', {})
    slot = permit.get('slot')
    _require(1 <= len(hand) <= 5 and type(slot) is int and 0 <= slot < len(hand),
             'selected baseline needs a bound physical target')
    ref = hand[slot]
    definition = bundle.kernel.cards[ref.card_id]
    card_id = definition.canonical_card_id or definition.card_id
    _require(definition.cost_kind is CardCostKind.NORMAL and definition.stamina_cost == 0
             and not definition.effects, 'selected baseline catalog cost or effects are unsupported')
    _gray_source_binding(bundle, complete_event)
    stages = journal.get('stages')
    _require(stages in (['selection_reserved', 'selection_dispatched'],
                        ['selection_reserved', 'selection_dispatched', 'selected_observation'])
             and journal.get('stage') == stages[-1]
             and journal.get('selection_ready') is True
             and type(journal.get('selection_count')) is int and journal['selection_count'] == 1
             and journal.get('business_submit_count', 0) == 0,
             'selected baseline lacks one unsubmitted recorded selection')
    _require(permit.get('accepted') is True and permit.get('phase') == 'selection_reserved'
             and permit.get('run_id') == bundle.run_id and permit.get('revision') == bundle.revision
             and permit.get('slot') == slot and permit.get('card_instance_id') == ref.instance_id
             and permit.get('card_id') == card_id and permit.get('maximum_selection_count') == 1
             and permit.get('maximum_submit_count') == 0
             and permit.get('submission_authorized') is False,
             'selected baseline reservation differs from the bound card')
    dispatch = [item for item in journal.get('selection_events', ())
                if item.get('phase') == 'execution_selection']
    _require(len(dispatch) == 1 and dispatch[0].get('attempted_slot') == slot
             and dispatch[0].get('accepted') is True and dispatch[0].get('click_dispatched') is True
             and dispatch[0].get('reason') == 'target_selection_dispatched',
             'selected baseline selection dispatch is unverified')
    certificates = journal.get('identity_certificate', ())
    source_a = journal.get('refresh_a', {})
    _require(_gray_hand_identity(bundle, certificates, source_a.get('identity_revalidation', ()))
             and tuple(map(tuple, complete_event.get('core', {}).get('families', ())))
                == tuple(tuple(cert.get('family', ())) for cert in certificates),
             'selected baseline complete hand identity differs')
    frames = evidence.get('frame_ids')
    previous_frames = [*complete_event.get('core', {}).get('frame_ids', ()),
                       *journal.get('refresh_a', {}).get('frame_ids', ()),
                       *journal.get('refresh_b', {}).get('frame_ids', ())]
    _require(_three_frames(frames) and not set(frames) & set(previous_frames),
             'selected baseline frames are not independently fresh')
    consensus = evidence.get('select_consensus') or {}
    selections = evidence.get('select_evidence', ())
    _require(evidence.get('geometry_stable') is True
             and consensus.get('accepted') is True and consensus.get('selected_slot') == slot
             and list(map(list, consensus.get('observations', ()))) == [[slot]]*3
             and len(selections) == 3 and all(
                 item.get('geometry_available') is True and item.get('geometry_ambiguous') is False
                 and item.get('ocr_slots') == [slot] and item.get('geometry_raised_slots') == [slot]
                 and item.get('resolved_slots') == [slot] for item in selections),
             'selected baseline lacks independent SELECT and raised geometry')
    original_box = permit.get('box', ())
    boxes = evidence.get('candidate_boxes', ())
    original = source_a.get('candidate_boxes', ())
    _require(len(original) == 3 and len(original[-1]) == len(hand)
             and list(original[-1][slot]) == list(original_box)
             and _gray_selected_hand_movement(boxes, original, slot, geometry_comparator),
             'selected baseline physical footprint differs')
    detail = evidence.get('detail_identity', {})
    titles = detail.get('observations', ())
    _require(detail.get('accepted') is True and detail.get('card_id') == card_id
             and len(titles) == 3 and [item.get('frame_id') for item in titles] == list(frames)
             and all(item.get('card_id') == card_id and item.get('roi') is not None for item in titles),
             'selected baseline lacks fresh complete identity')
    resource = evidence.get('resource_transaction', {})
    _require(resource.get('accepted') is True and resource.get('selected_slot') == slot
             and resource.get('frame_ids') == list(frames)
             and all(type(resource.get(name)) is int and resource[name] == 0
                     for name in ('displayed_effective_cost', 'stamina_delta', 'shield_delta'))
             and resource.get('normalized_stamina') == resource.get('preview_stamina') == bundle.state.stamina
             and resource.get('normalized_shield') == resource.get('preview_shield') == bundle.state.genki,
             'selected baseline zero-cost resource transaction changed')
    observations = resource.get('observations', {})
    absolutes = observations.get('absolutes', {})
    costs = observations.get('effective_cost', ())
    _require(absolutes.get('accepted') is True and absolutes.get('stamina') == bundle.state.stamina
             and absolutes.get('shield') == bundle.state.genki
             and len(absolutes.get('observations', ())) == 3
             and all(row.get(key, {}).get('accepted') is True and row[key].get('value') == value
                     for row in absolutes['observations']
                     for key, value in (('stamina', bundle.state.stamina), ('shield', bundle.state.genki)))
             and len(costs) == 3 and all(item.get('accepted') is True
                 and type(item.get('value')) is int and item['value'] == 0 for item in costs),
             'selected baseline needs three actual zero costs and absolute resources')
    confidence = resource.get('confidence')
    _require(type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1,
             'selected baseline resource confidence is invalid')
    return dict(accepted=True, stamina=bundle.state.stamina, shield=bundle.state.genki,
        confidence=confidence, reason='live_resource_baseline_selected_zero_cost_unanimous',
        frame_ids=list(frames), candidate_boxes=copy.deepcopy(boxes), geometry_stable=True,
        select_consensus=copy.deepcopy(consensus), select_evidence=copy.deepcopy(selections),
        unselected_geometry_authorized=False, selected_geometry_authorized=True, selected_slot=slot,
        resource_observations=copy.deepcopy(absolutes['observations']),
        detail_identity=copy.deepcopy(detail), resource_transaction=copy.deepcopy(resource),
        submission_authorized=False)


def selected_gray_playability(bundle, *, complete_event, journal, baseline,
                             event, candidate, certificate, geometry_comparator):
    """Prove this selected card usable, leaving policy and input gates intact.

    NORMAL/zero-cost/no-effects cards are playable under the existing kernel;
    the selected preview supplies independent geometry, identity and actual
    cost. This does not mutate the detector label or its clickable flag.
    The host subsequently applies its usual 1000 ms and safety checks.
    """
    selected_zero_cost_baseline(bundle, complete_event=complete_event, journal=journal,
                               evidence=baseline, geometry_comparator=geometry_comparator)
    slot = journal['permit']['slot']
    ref = bundle.state.deck.hand[slot]
    definition = bundle.kernel.cards[ref.card_id]
    card_id = definition.canonical_card_id or definition.card_id
    _require(card_id == '23' and definition.card_type == 'trouble',
             'selected gray playability has only been verified for trouble card 23')
    legal = bundle.kernel.legal_actions(bundle.state, plan_type=PlanType.SENSE,
                                       decision_kind=DecisionKind.LIVE)
    _require(any(action.kind is ActionKind.PLAY_CARD and action.card_instance_id == ref.instance_id
                 for action in legal), 'selected gray card is not kernel legal')
    _require(candidate.slot == slot and candidate.visible is True and candidate.clickable is False
             and candidate.detector_label == 'hif_gray_observation'
             and certificate.get('slot') == slot and certificate.get('card_id') == card_id
             and certificate.get('upgraded') is ref.upgraded
             and tuple(certificate.get('family', ())) == (card_id,),
             'selected gray target or full identity differs')
    frames = event.get('frame_ids')
    old_frames = [*baseline['frame_ids'], *journal['refresh_a']['frame_ids'],
                  *journal.get('refresh_b', {}).get('frame_ids', ())]
    _require(event.get('phase') in ('authoritative_refresh_a', 'authoritative_refresh_b')
             and event.get('expected_selected_slot') == slot
             and _three_frames(frames) and not set(frames) & set(old_frames)
             and event.get('generation') == journal['permit']['generation'],
             'selected gray refresh is not independent or core changed')
    consensus = event.get('select_consensus') or {}
    select = event.get('select_evidence', ())
    boxes = event.get('candidate_boxes', ())
    reference = baseline.get('candidate_boxes', ())
    _require(consensus.get('accepted') is True and consensus.get('selected_slot') == slot
             and list(map(list, consensus.get('observations', ()))) == [[slot]]*3
             and len(select) == 3 and all(item.get('geometry_available') is True
                 and item.get('geometry_ambiguous') is False and item.get('ocr_slots') == [slot]
                 and item.get('geometry_raised_slots') == [slot] and item.get('resolved_slots') == [slot]
                 for item in select)
             and len(boxes) == 3 and len(boxes[-1]) == len(bundle.state.deck.hand)
             and _complete_gray_hand_boxes((candidate.box,))
             and geometry_comparator((candidate,), (SimpleNamespace(box=tuple(boxes[-1][slot])),))
             and _same_selected_geometry(boxes, reference[-1], geometry_comparator)
             and _gray_selected_hand_movement(boxes, journal['refresh_a']['candidate_boxes'],
                                             slot, geometry_comparator),
             'selected gray SELECT or full geometry is not unanimous')
    identity = event.get('identity_revalidation', ())
    _require(_gray_hand_identity(bundle, journal.get('identity_certificate', ()), identity),
             'selected gray identity must be independently recognized')
    resource = event.get('resource_transaction', {})
    _require(resource.get('accepted') is True and resource.get('selected_slot') == slot
             and resource.get('card_id') == card_id and resource.get('frame_ids') == list(frames)
             and all(type(resource.get(name)) is int and resource[name] == 0
                     for name in ('displayed_effective_cost', 'stamina_delta', 'shield_delta'))
             and resource.get('normalized_stamina') == resource.get('preview_stamina') == bundle.state.stamina
             and resource.get('normalized_shield') == resource.get('preview_shield') == bundle.state.genki,
             'selected gray cost or resources changed')
    return dict(accepted=True, reason='verified_zero_cost_selected_gray',
        run_id=bundle.run_id, revision=bundle.revision, card_id=card_id,
        card_instance_id=ref.instance_id, slot=slot, frame_ids=list(frames),
        baseline_frame_ids=list(baseline['frame_ids']), submission_authorized=False)


def ordinary_nullified_resource_preview(bundle, slot):
    """Interpret a normal card face separately from verified nullified payment.

    906 displays the non-nullified cost on the physical card while both HUD
    resources stay unchanged. This is a model contract, never an observation:
    the host must match its card-face cost and both resource deltas to fresh
    three-frame evidence before accepting the transaction.
    """
    state, kernel = bundle.state, bundle.kernel
    _require(type(slot) is int and 0 <= slot < len(state.deck.hand),
             'ordinary nullified preview slot is invalid')
    if type(state.nullify_cost_cards) is not int or state.nullify_cost_cards <= 0:
        return None
    ref = state.deck.hand[slot]
    definition = kernel.cards[ref.card_id]
    if (definition.card_type not in {'mental', 'active'}
            or definition.cost_kind not in {CardCostKind.NORMAL, CardCostKind.STAMINA}
            or definition.persistent_effects
            or any(effect.kind not in {EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION}
                   for effect in definition.effects)
            or kernel._effective_stamina_cost(state, ref, definition) != 0):
        return None
    legal = kernel.legal_actions(state, plan_type=PlanType.SENSE, decision_kind=DecisionKind.LIVE)
    if not any(action.kind is ActionKind.PLAY_CARD and action.card_instance_id == ref.instance_id
               for action in legal):
        return None
    displayed = kernel._effective_stamina_cost(replace(state, nullify_cost_cards=0), ref, definition)
    return dict(source='native_hif_card_effect_preview', run_id=bundle.run_id,
        revision=bundle.revision, card_id=definition.canonical_card_id or definition.card_id,
        selected_slot=slot, baseline_stamina=state.stamina, baseline_shield=state.genki,
        effective_cost=0, preview_stamina=state.stamina, preview_shield=state.genki,
        preview_score=state.score, displayed_card_cost=displayed, nullify_cost_cards=state.nullify_cost_cards)


def selected_preview_matches_temporary_identity(bundle, policy, preview, certificate, slot):
    """Keep the permanent preview ID while proving its observed temporary form."""
    try:
        if (bundle is None or policy.bundle != bundle or type(slot) is not int
                or not 0 <= slot < len(bundle.state.deck.hand)
                or (preview.get('run_id'), preview.get('revision')) != (bundle.run_id,bundle.revision)
                or preview.get('source') != 'native_hif_card_effect_preview'
                or preview.get('selected_slot') != slot or certificate.get('slot') != slot
                or certificate.get('upgraded') is not True):
            return False
        ref = bundle.state.deck.hand[slot]
        action = policy.response.action if policy.response else None
        if (ref.card_id != '288' or preview.get('card_id') != ref.card_id
                or action is None or action.kind is not ActionKind.PLAY_CARD
                or action.source_slot != slot or action.card_instance_id != ref.instance_id
                or not {'288','289'} <= set(certificate.get('family', ()))
                or policy._current_support_definition() is None):
            return False
        return certificate.get('card_id') == policy._opening_support_visible_id(ref) == '289'
    except (ValueError, KeyError, TypeError, AttributeError):
        return False


def _complete_recoverable_support(receipt, card_id):
    from .high_score.support_master import SOURCE
    return ((card_id in ('35', '818', '709') and receipt.get('enhancement_count') == 2)
        or (receipt.get('effect_source') == SOURCE and receipt.get('enhancement_count') in (2, 3)))


def recovered_selected_support_visible_id(bundle, *, complete_event, journal):
    """Identify which selected detail needs a fresh proof, never grant input."""
    from .high_score.contracts import GameAction
    from .high_score.live_policy import BoundHifOpeningPolicy

    raw = journal.get('refresh_a', {}).get('proposal', {}).get('candidate_traces', {}).get(
        'high_score', {}).get('action', {})
    _require(raw.get('kind') == 'PLAY_CARD', 'fresh selected support lacks the original action')
    action = GameAction(**{**raw, 'kind': ActionKind(raw['kind'])})
    action.validate()
    _require(type(action.source_slot) is int and 0 <= action.source_slot < len(bundle.state.deck.hand),
             'fresh selected support target is outside the bound hand')
    ref = bundle.state.deck.hand[action.source_slot]
    policy = BoundHifOpeningPolicy(bundle, action_order=complete_event['core']['fields'][5])
    policy.response = SimpleNamespace(action=action)
    _require(action.card_instance_id == ref.instance_id,
             'fresh selected support differs from the bound instance')
    receipt = journal.get('selected_skill_support')
    if receipt is not None:
        # Reverify the complete old receipt against the exact unsubmitted
        # action/state. It identifies what must be freshly observed below.
        policy.bind_selected_skill_support(receipt)
        definition = policy._current_support_definition()
        if receipt.get('enhancement_count') != 1 or ref.upgraded:
            if ref.upgraded and _complete_recoverable_support(receipt, ref.card_id):
                _require(definition is not None, 'selected recovery lacks the verified support definition')
                return ref.card_id
            # Existing plus-to-plus-plus recovery keeps its prior path.
            return None
        _require(definition is not None and definition.upgrade_card_id is not None,
                 'selected recovery requires a verified one-level base upgrade')
        return definition.upgrade_card_id
    pending = [step.get('receipt') for step in journal.get('selection_events', ())
               if step.get('phase') == 'hif_selected_skill_support']
    pending_reasons = {'818':'selected extra enhancement is outside the verified slice',
        '35':'selected support complete body or printed cost is unresolved',
        '709':'selected support complete body or printed cost is unresolved'}
    if ref.card_id in pending_reasons and ref.upgraded and pending:
        from .hif_skill_support import verify_selected_skill_support_observations

        _require(len(pending) == 1 and isinstance(pending[0], dict),
                 'selected recovery has ambiguous unverified support details')
        previous = pending[0]
        _require(previous.get('accepted') is False and previous.get('effects_verified') is False
                 and previous.get('submission_authorized') is False
                 and previous.get('reason') == pending_reasons[ref.card_id]
                 and previous.get('card_id') == ref.card_id
                 and previous.get('card_instance_id') == ref.instance_id
                 and previous.get('selected_slot') == action.source_slot
                 and (previous.get('run_id'),previous.get('revision')) == (bundle.run_id,bundle.revision)
                 and json.dumps(previous.get('bound_state'),sort_keys=True)
                     == json.dumps(asdict(bundle.state),sort_keys=True),
                 'unverified support title differs from the original selected instance')
        title = verify_selected_skill_support_observations(previous.get('observations', ()),
            bundle=bundle,selected_slot=action.source_slot,
            selection_evidence=previous.get('selection_evidence', {}))
        _require(title.get('title_verified') is True and title.get('support_required') is True
                 and title.get('plus_count') == 2,
                 'unverified support source lacks its complete selected title')
        if ref.card_id in ('35','709'):
            from .hif_skill_support import _qualified, _original_support_body_complete

            # 1695's raw body includes a separator decoration before the first
            # effect. It may request a fresh full read, never bind filtered text.
            # A contradictory otherwise complete 942 body retains its rejection.
            _require(title.get('frame_ids') == previous.get('frame_ids')
                     and title.get('raw_title') == previous.get('raw_title')
                         == ('ハイタッチ++' if ref.card_id == '35' else '大胆不敵++')
                     and previous.get('definition_id') == ref.card_id,
                     'unverified support source lacks its unresolved header body')
            for observation in previous['observations']:
                x,y,w,_h = observation['panel_box']
                raw_rows = observation.get('raw_body_rows' if ref.card_id == '35' else 'body_rows', ())
                header = [line for line in raw_rows if line['box'][1] < y+74]
                labels = [line for line in header if line['text'] in ('Active','Mental')]
                extra = [line for line in header if line not in labels]
                _require(len(labels) == 1 and _qualified(labels,(x+12,y+50,w-24,24))
                         and abs(labels[0]['box'][0]+labels[0]['box'][2]/2-(x+w/2)) <= 8
                         and 1 <= len(extra) <= 2 and all(
                             y+50 <= line['box'][1] and line['box'][1]+line['box'][3] <= y+74
                             and 0 < line['box'][2] <= 14 and 0 < line['box'][3] <= 14
                             and (x+12 <= line['box'][0] <= x+32
                                  or x+w-(40 if ref.card_id == '709' else 36)
                                     <= line['box'][0] <= x+w-12-line['box'][2])
                             for line in extra)
                         and _original_support_body_complete(observation['title_rows'],
                             observation['title_roi'],[line for line in raw_rows if line not in extra],
                             observation['body_roi'],observation['panel_box']),
                         'unverified support source has incomplete or contradictory effect text')
        # The failed body's confidence is deliberately not promoted. This ID
        # only dispatches a new complete detail read before resource recovery.
        return ref.card_id
    # The older breath recovery predates selected receipts; retain its
    # independently verified opening support route without broadening it.
    if ref.card_id == '288' and policy._opening_support_visible_id(ref) == '289':
        return '289'
    return None


def recovered_selected_support_journal(bundle, *, complete_event, journal, evidence):
    """Bind fresh complete support only to this resource recovery copy."""
    from .high_score.contracts import GameAction
    from .high_score.live_policy import BoundHifOpeningPolicy

    receipt = evidence.get('selected_skill_support')
    if receipt is None:
        return journal
    visible_id = recovered_selected_support_visible_id(bundle,
        complete_event=complete_event, journal=journal)
    _require(visible_id is not None, 'fresh selected support lacks the verified opening instance')
    raw = journal['refresh_a']['proposal']['candidate_traces']['high_score']['action']
    policy = BoundHifOpeningPolicy(bundle, action_order=complete_event['core']['fields'][5])
    policy.response = SimpleNamespace(action=GameAction(**{**raw, 'kind': ActionKind(raw['kind'])}))
    _require(isinstance(receipt, dict) and receipt.get('accepted') is True
             and list(receipt.get('frame_ids', ())) == list(evidence.get('frame_ids', ()))
             and all(receipt.get('selection_evidence', {}).get(key) == evidence.get(key)
                     for key in ('geometry_stable', 'select_consensus', 'select_evidence')),
             'fresh selected support differs from the current recovery frames or SELECT')
    policy.bind_selected_skill_support(receipt)
    definition = policy._current_support_definition()
    ref = bundle.state.deck.hand[raw['source_slot']]
    _require((receipt.get('enhancement_count') == 1 and definition.upgrade_card_id == visible_id)
             or (ref.upgraded and _complete_recoverable_support(receipt, ref.card_id)
                 and definition.card_id == visible_id == ref.card_id),
             'fresh selected support changed the verified upgrade level')
    return dict(journal, selected_skill_support=copy.deepcopy(receipt))


def recovered_selected_support_hand_identity(bundle, *, complete_event, journal, baseline,
                                            frame_ids, slot):
    """Consume this observation's full receipt while keeping permanent identity."""
    _require(isinstance(baseline, dict) and baseline.get('accepted') is True
             and baseline.get('selected_geometry_authorized') is True
             and baseline.get('selected_slot') == slot
             and list(baseline.get('frame_ids', ())) == list(frame_ids),
             'selected temporary identity lacks this observation baseline')
    if baseline.get('single_plus_estimate') is not None:
        refreshed = dict(journal, single_plus_estimate=baseline['single_plus_estimate'])
        policy = _ordinary_selected_preview_policy(bundle, complete_event, refreshed)
        receipt = policy.selected_single_plus_estimate
        visible_id = recovered_selected_ordinary_plus_visible_id(bundle,
            complete_event=complete_event, journal=refreshed)
        _require(visible_id is not None and receipt['selected_slot'] == slot
                 and receipt['frame_ids'] == list(frame_ids)
                 and baseline.get('detail_identity', {}).get('accepted') is True
                 and baseline['detail_identity'].get('card_id') == visible_id,
                 'selected ordinary-plus identity differs from this fresh baseline')
        return dict(accepted=True, source='fresh_selected_ordinary_plus_baseline',
            card_id=receipt['card_id'], card_instance_id=receipt['card_instance_id'],
            visible_card_id=visible_id, slot=slot, run_id=bundle.run_id, revision=bundle.revision,
            frame_ids=list(frame_ids), effects_verified=False, submission_authorized=False)
    _require(baseline.get('selected_skill_support') is not None,
             'selected temporary identity lacks this observation baseline')
    refreshed = recovered_selected_support_journal(bundle, complete_event=complete_event,
        journal=journal, evidence=baseline)
    receipt = refreshed['selected_skill_support']
    _require(receipt.get('selected_slot') == slot,
             'selected temporary identity changed its physical slot')
    ref = bundle.state.deck.hand[slot]
    return dict(accepted=True, source='fresh_selected_resource_baseline',
        card_id=receipt['card_id'], card_instance_id=ref.instance_id,
        visible_card_id=recovered_selected_support_visible_id(bundle,
            complete_event=complete_event, journal=refreshed),
        slot=slot, run_id=bundle.run_id, revision=bundle.revision,
        frame_ids=list(frame_ids), effects_verified=True, submission_authorized=False)


def _ordinary_selected_preview_policy(bundle, complete_event, journal):
    """Bind the original selected entity for the existing preview consumers."""
    from .high_score.contracts import GameAction
    from .high_score.live_policy import BoundHifOpeningPolicy

    raw = journal.get('refresh_a', {}).get('proposal', {}).get('candidate_traces', {}).get(
        'high_score', {}).get('action', {})
    _require(isinstance(raw, dict) and raw.get('kind') == 'PLAY_CARD',
             'ordinary selected resource preview lacks its recorded action')
    action = GameAction(**{**raw, 'kind': ActionKind(raw['kind'])})
    action.validate()
    _require(type(action.source_slot) is int and 0 <= action.source_slot < len(bundle.state.deck.hand)
             and action.card_instance_id == bundle.state.deck.hand[action.source_slot].instance_id,
             'ordinary selected resource preview action differs from the bound hand')
    policy = BoundHifOpeningPolicy(bundle, action_order=complete_event['core']['fields'][5])
    policy.response = SimpleNamespace(action=action)
    if journal.get('selected_skill_support') is not None:
        # The native adapter has retained a previously observed proof. Replay
        # it against this exact prestate for resource normalization only; the
        # caller still rereads selected details before a new business input.
        policy.bind_selected_skill_support(journal['selected_skill_support'])
    if journal.get('single_plus_estimate') is not None:
        policy.bind_selected_single_plus_estimate(journal['single_plus_estimate'], require_submitted=False)
    return policy


def ordinary_selected_resource_preview(bundle, *, complete_event, journal):
    """Recompute the typed contract without choosing a new card or observing it."""
    policy = _ordinary_selected_preview_policy(bundle, complete_event, journal)
    # The policy already limits effect families and requires deterministic
    # resources from one bound card transition. A nonzero payment can coexist
    # with recovery (1005: pay 4, then recover to 49); it is not a gray-card case.
    return policy.selected_resource_preview(policy.response.action.source_slot)


def recovered_selected_ordinary_plus_visible_id(bundle, *, complete_event, journal):
    """A rechecked title estimate identifies the required fresh upgrade title."""
    if journal.get('single_plus_estimate') is None:
        return None
    policy = _ordinary_selected_preview_policy(bundle, complete_event, journal)
    receipt = policy.selected_single_plus_estimate
    return (receipt['projected_card_id'] if receipt.get('ordinary_plus_projection') is True else None)


def ordinary_selected_observed_score_preview(bundle, *, complete_event, journal, live, fields):
    """Reuse ordinary-card HUD observation before the initial recovery binding.

    The same policy consumer used by A/B validates every public field and the
    current resource transaction. Its observed forecast never replaces OCR or
    becomes the already committed score. No search runs here.
    """
    policy = _ordinary_selected_preview_policy(bundle, complete_event, journal)
    transaction, score = fields.get('resource_transaction', {}), fields.get('score', {})
    action = policy.response.action
    definition = bundle.kernel.card_definition(bundle.state,bundle.state.deck.hand[action.source_slot])
    _require(journal.get('ordinary_safe_preview') is True
             and live.snapshot_id in transaction.get('frame_ids', ())
             and score.get('accepted') is True
             and type(score.get('confidence')) in (int, float)
             and .95 <= score['confidence'] <= 1,
             'ordinary initial score preview lacks fresh accepted OCR')
    _require(transaction.get('accepted') is True
             and transaction.get('selected_slot') == action.source_slot
             and len(set(transaction.get('frame_ids', ()))) == 3
             and transaction.get('normalized_stamina') == bundle.state.stamina
             and transaction.get('normalized_shield') == bundle.state.genki
             and score['value'] == live.score,
             'ordinary initial score preview is outside the existing observation scope')
    ordinary = (definition.cost_kind is CardCostKind.NORMAL
        and any(effect.kind is EffectKind.SCORE for effect in definition.effects)
        and all(effect.kind in (EffectKind.SCORE, EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION)
                for effect in definition.effects))
    # The separate typed consumer requires the verified multi-plus title and
    # exact current resource/model contract before the full public-state check.
    typed = not ordinary and (policy.bind_single_plus_typed_score_preview(transaction, score)
        or policy._bind_typed_single_precision_display(transaction, score))
    _require(ordinary or typed,
             'ordinary initial score preview is outside the existing observation scope')
    policy.prepare_state(live, fields)
    observed = policy.observed_normal_score_preview
    _require(observed is not None and observed['score'] == score['value'],
             'ordinary initial score preview is outside the existing observation scope')
    return dict(source='ordinary_selected_hud_observation',run_id=bundle.run_id,
        revision=bundle.revision,selected_slot=policy.response.action.source_slot,
        card_instance_id=policy.response.action.card_instance_id,
        frame_ids=list(transaction['frame_ids']),observed_score=score['value'],
        baseline_score=bundle.state.score,confidence=score['confidence'],
        model_preview_score=(transaction.get('card_effect_preview') or {}).get('preview_score'),
        submission_authorized=False)


def _ordinary_observed_payment(bundle, definition, resource):
    """Reuse existing ordinary-card observed-payment policy families."""
    from .high_score.live_policy import (
        status_only_resource_effects,
        hand_exchange_resource_effects,
        generated_draw_resource_effects,
    )

    kinds = {effect.kind for effect in definition.effects}
    score_status = (EffectKind.SCORE in kinds and kinds <= {
        EffectKind.SCORE, EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION})
    generated_draw = generated_draw_resource_effects(definition.effects)
    # Adding a declared card to the deck does not play it or change HUD
    # resources. These status/action effects therefore retain the ordinary
    # observed-payment contract even when the card has no DRAW effect.
    generated_status = (EffectKind.ADD_CARD_TO_DECK in kinds and kinds <= {
        EffectKind.CONCENTRATION, EffectKind.GOOD_CONDITION, EffectKind.EXTRA_ACTION,
        EffectKind.ADD_CARD_TO_DECK} and all(
            isinstance(getattr(effect, 'card_id', None), str) and bool(effect.card_id)
            for effect in definition.effects if effect.kind is EffectKind.ADD_CARD_TO_DECK))
    status_only = status_only_resource_effects(definition.effects)
    _require(definition.cost_kind is CardCostKind.NORMAL
             and (score_status or generated_draw or generated_status or status_only
                  or hand_exchange_resource_effects(definition)),
             'ordinary selected card has no supported resource preview')
    cost = resource.get('displayed_effective_cost')
    state = bundle.state
    _require(type(cost) is int and 0 <= cost <= state.stamina + state.genki
             and resource.get('card_effect_preview') is None
             and all(resource.get('conservation', {}).get(name) is True for name in
                     ('effective_cost', 'stamina', 'shield', 'shield_first_split')),
             'ordinary observed payment lacks complete conservation')
    # No modeled score is invented here. Fresh A/B bind the independently read
    # score forecast through the existing ordinary-card policy before submit.
    return dict(effective_cost=cost,
        preview_stamina=state.stamina-max(cost-state.genki, 0),
        preview_shield=max(state.genki-cost, 0))


def _ordinary_selected_rule_binding(bundle, event):
    from .high_score.hif_rule_progress import _baseline

    receipt = event.get('rule_preinput' if event.get('verification_scope') == 'rule_maintained_preinput_v1'
                        else 'rule_progress')
    _require(isinstance(receipt, dict) and all(
        event.get(key) == receipt.get(key) for key in (
            'accepted', 'strategy_state_bound', 'run_id', 'revision', 'verification_scope',
            'checks', 'core', 'expected_state')) and not event.get('components'),
        'ordinary selected rule receipt differs from its verified envelope')
    _baseline(bundle, receipt)


def ordinary_selected_zero_cost_baseline(bundle, *, complete_event, journal, evidence,
                                        geometry_comparator, opening_run=None,
                                        round_context=None, data_root=None):
    """Recover a selected card with its existing typed resource preview.

    The card face and both HUD deltas must match fresh three-frame evidence.
    Cards can pay and recover stamina; normalization still preserves the
    complete pre-input state until the separate business submission settles.
    """
    journal = recovered_selected_support_journal(bundle, complete_event=complete_event,
        journal=journal, evidence=evidence)
    projected_id = recovered_selected_ordinary_plus_visible_id(bundle,
        complete_event=complete_event, journal=journal)
    if projected_id is not None:
        fresh = evidence.get('single_plus_estimate')
        _require(isinstance(fresh, dict) and fresh.get('frame_ids') == list(evidence.get('frame_ids', ())),
                 'ordinary selected projection lacks this fresh title batch')
        journal = dict(journal, single_plus_estimate=fresh)
        _require(recovered_selected_ordinary_plus_visible_id(bundle,
            complete_event=complete_event, journal=journal) == projected_id,
            'ordinary selected projection title changed')
    hand = bundle.state.deck.hand
    a = journal.get('refresh_a', {})
    action = a.get('proposal', {}).get('candidate_traces', {}).get('high_score', {}).get('action', {})
    slot = action.get('source_slot')
    _require(1 <= len(hand) <= 5 and type(slot) is int and 0 <= slot < len(hand)
             and action.get('kind') == 'PLAY_CARD' and action.get('drink_id') is None
             and action.get('target_id') is None, 'ordinary selected action is unbound')
    ref = hand[slot]
    definition = bundle.kernel.cards[ref.card_id]
    card_id = definition.canonical_card_id or definition.card_id
    contract = ordinary_selected_resource_preview(bundle, complete_event=complete_event, journal=journal)
    _require(action.get('card_instance_id') == ref.instance_id,
             'ordinary selected action instance differs from the bound hand')
    _require(complete_event.get('accepted') is True
             and complete_event.get('run_id') == bundle.run_id
             and complete_event.get('revision') == bundle.revision,
             'ordinary selected baseline lacks complete binding')
    if complete_event.get('event') == 'live_opening_state':
        from .hif_visual import HifLiveRunContext
        from .high_score.opening import assemble_hif_opening

        _require(isinstance(opening_run, HifLiveRunContext) and data_root is not None
                 and (opening_run.run_id, opening_run.revision) == (bundle.run_id, bundle.revision)
                 and complete_event.get('executed') is False
                 and complete_event.get('execution_submit_count') == 0
                 and complete_event.get('strategy_state_bound') is True,
                 'ordinary selected opening lacks its original bound run')
        rebuilt = assemble_hif_opening(opening_run, core=complete_event['core'],
            components=complete_event['components'], hand_titles=complete_event['hand_titles'],
            data_root=data_root, round_context=round_context,
            opening_skill_support=complete_event.get('opening_skill_support', ()),
            extended_hand_identity=complete_event.get('extended_hand_identity'),
            confirmed_deck_baseline=complete_event.get('confirmed_deck_baseline'))
        _require(replace(rebuilt, kernel=bundle.kernel) == bundle
                 and type(rebuilt.kernel) is type(bundle.kernel)
                 and all(getattr(rebuilt.kernel, key) == getattr(bundle.kernel, key)
                         for key in ('cards', 'drinks', 'items', 'memory_abilities', 'ruleset_id'))
                 and json.dumps(asdict(rebuilt.state), sort_keys=True)
                     == json.dumps(complete_event.get('state'), sort_keys=True),
                 'ordinary selected opening differs from its complete observations')
    elif complete_event.get('verification_scope') in (
            'rule_maintained_information_boundary_v1', 'rule_maintained_preinput_v1'):
        _ordinary_selected_rule_binding(bundle, complete_event)
    else:
        _require(isinstance(complete_event.get('checks'), dict)
                 and {'core', 'hand_identity', 'all_card_zones', 'drinks', 'turns'} <= set(complete_event['checks'])
                 and all(value is True for value in complete_event['checks'].values()),
                 'ordinary selected baseline lacks complete binding')
    _require(journal.get('ordinary_safe_preview') is True and 'permit' not in journal
             and journal.get('selection_ready') is True
             and type(journal.get('selection_count')) is int and journal['selection_count'] == 1
             and journal.get('business_submit_count', 0) == 0
             and journal.get('stages') in (['selection_reserved', 'selection_dispatched'],
                 ['selection_reserved', 'selection_dispatched', 'selected_observation'], ['selection_dispatched'])
             and journal.get('stage') == journal['stages'][-1],
             'ordinary selected recovery lacks exactly one unsubmitted selection')
    if journal['stages'] == ['selection_dispatched']:
        _require(journal.get('provenance') == 'completed_native_selection_report'
                 and journal.get('source_event') in ('live_selection_stop', 'live_decision_observe')
                 and isinstance(journal.get('source_record'), str) and journal['source_record'],
                 'ordinary completed selection report lacks audited provenance')
    dispatch = [item for item in journal.get('selection_events', ()) if item.get('phase') == 'execution_selection']
    _require(len(dispatch) == 1 and dispatch[0].get('accepted') is True
             and dispatch[0].get('attempted_slot') == slot and dispatch[0].get('click_dispatched') is True
             and dispatch[0].get('reason') == 'target_selection_dispatched',
             'ordinary selected dispatch differs from the reserved target')
    generation = dict(turn=bundle.state.turn, turns_remaining=bundle.state.turns_remaining,
        stamina=bundle.state.stamina, shield=bundle.state.genki, score=bundle.state.score,
        score_multiplier=bundle.state.score_multiplier, hand_count=len(hand), plan='sense',
        action_order=list(complete_event.get('core', {}).get('fields', (None,)*6)[5]))
    safe = a.get('safe_decision', {})
    _require(a.get('accepted') is True and a.get('phase') == 'authoritative_refresh_a'
             and a.get('expected_selected_slot') is None and a.get('unselected_geometry_authorized') is True
             and a.get('generation') == generation and safe.get('status') == 'ALLOW'
             and safe.get('has_action_intent') is True
             and safe.get('action') == dict(kind='PLAY_CARD', card_slot=slot),
             'ordinary selected source A was not independently approved')
    certificates = journal.get('identity_certificate', ())
    _require(len(certificates) == len(hand) and all(
        cert.get('slot') == i and cert.get('card_id') == (bundle.kernel.cards[card.card_id].canonical_card_id or card.card_id)
        and cert.get('upgraded') is card.upgraded and cert['card_id'] in cert.get('family', ())
        for i, (card, cert) in enumerate(zip(hand, certificates, strict=True))),
        'ordinary selected hand certificates differ from the binding')
    frames = evidence.get('frame_ids')
    previous_frames = [*complete_event.get('core', {}).get('frame_ids', ()),
                       *a.get('frame_ids', ()), *journal.get('refresh_b', {}).get('frame_ids', ())]
    _require(_three_frames(frames) and not set(frames) & set(previous_frames),
             'ordinary selected baseline frames are not independently fresh')
    consensus = evidence.get('select_consensus') or {}
    selections = evidence.get('select_evidence', ())
    _require(evidence.get('geometry_stable') is True and consensus.get('accepted') is True
             and consensus.get('selected_slot') == slot
             and list(map(list, consensus.get('observations', ()))) == [[slot]]*3
             and len(selections) == 3 and all(item.get('geometry_available') is True
                 and item.get('geometry_ambiguous') is False and item.get('ocr_slots') == [slot]
                 and item.get('geometry_raised_slots') == [slot] and item.get('resolved_slots') == [slot]
                 for item in selections), 'ordinary selected baseline lacks independent SELECT and raised geometry')
    boxes = evidence.get('candidate_boxes', ())
    old_boxes = a.get('candidate_boxes', ())
    _require(len(boxes) == len(old_boxes) == 3
             and all(len(row) == len(hand) and all(len(box) == 4 for box in row) for row in (*boxes, *old_boxes))
             and tuple(boxes[-1][slot][2:]) == (190,250)
             and abs(boxes[-1][slot][0]-old_boxes[-1][slot][0]) <= 8
             and 16 <= old_boxes[-1][slot][1]-boxes[-1][slot][1] <= 36
             and callable(geometry_comparator) and all(geometry_comparator(
                 tuple(SimpleNamespace(box=box) for box in boxes[-1]),
                 tuple(SimpleNamespace(box=box) for box in row)) for row in boxes),
             'ordinary selected baseline physical hand differs')
    detail = evidence.get('detail_identity', {})
    titles = detail.get('observations', ())
    visible_card_id = (recovered_selected_support_visible_id(bundle,
        complete_event=complete_event, journal=journal)
        if evidence.get('selected_skill_support') is not None else projected_id or card_id)
    _require(detail.get('accepted') is True and detail.get('card_id') == visible_card_id and len(titles) == 3
             and [item.get('frame_id') for item in titles] == list(frames)
             and all(item.get('card_id') == visible_card_id and item.get('roi') is not None for item in titles),
             'ordinary selected baseline lacks fresh complete target identity')
    resource = evidence.get('resource_transaction', {})
    payment = contract if contract is not None else _ordinary_observed_payment(bundle, definition, resource)
    displayed_cost = payment.get('displayed_card_cost', payment['effective_cost'])
    preview_stamina, preview_shield = payment['preview_stamina'], payment['preview_shield']
    resource_deltas = dict(effective_payment_cost=payment['effective_cost'],
        stamina_delta=preview_stamina-bundle.state.stamina, shield_delta=preview_shield-bundle.state.genki)
    _require(resource.get('accepted') is True and resource.get('selected_slot') == slot
             and resource.get('frame_ids') == list(frames)
             and resource.get('card_effect_preview') == contract
             and resource.get('card_face_cost') == resource.get('displayed_effective_cost')
                == displayed_cost
             and all(type(resource.get(name)) is int and resource[name] == value
                     for name, value in resource_deltas.items())
             and resource.get('normalized_stamina') == bundle.state.stamina
             and resource.get('normalized_shield') == bundle.state.genki
             and resource.get('preview_stamina') == preview_stamina
             and resource.get('preview_shield') == preview_shield,
             'ordinary selected card face, payment, or typed resource preview changed')
    observed = resource.get('observations', {})
    absolutes, costs = observed.get('absolutes', {}), observed.get('effective_cost', ())
    _require(absolutes.get('accepted') is True and absolutes.get('stamina') == preview_stamina
             and absolutes.get('shield') == preview_shield and len(absolutes.get('observations', ())) == 3
             and all(row.get(key, {}).get('accepted') is True and row[key].get('value') == value
                     for row in absolutes['observations']
                     for key,value in (('stamina',preview_stamina),('shield',preview_shield)))
             and len(costs) == 3 and all(item.get('accepted') is True and type(item.get('value')) is int
                                      and item['value'] == displayed_cost for item in costs),
             'ordinary selected baseline needs three actual card costs and matching preview resources')
    confidence = resource.get('confidence')
    _require(type(confidence) in (int,float) and math.isfinite(confidence) and 0 <= confidence <= 1,
             'ordinary selected resource confidence is invalid')
    return dict(accepted=True, stamina=bundle.state.stamina, shield=bundle.state.genki,
        confidence=confidence, reason='live_resource_baseline_ordinary_selected_preview_unanimous',
        frame_ids=list(frames), candidate_boxes=copy.deepcopy(boxes), geometry_stable=True,
        select_consensus=copy.deepcopy(consensus), select_evidence=copy.deepcopy(selections),
        unselected_geometry_authorized=False, selected_geometry_authorized=True, selected_slot=slot,
        resource_observations=copy.deepcopy(absolutes['observations']), detail_identity=copy.deepcopy(detail),
        resource_transaction=copy.deepcopy(resource), submission_authorized=False,
        **({'selected_skill_support': copy.deepcopy(evidence['selected_skill_support'])}
           if evidence.get('selected_skill_support') is not None else {}),
        **({'single_plus_estimate': copy.deepcopy(evidence['single_plus_estimate'])}
           if projected_id is not None else {}))
