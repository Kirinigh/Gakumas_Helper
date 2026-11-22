"""D-047 title evidence: estimate Support enhancement with the ordinary plus card.

Permanent customization remains bound to this run's inventory. This receipt
does not verify temporary effects, grant input permission, or change the model.
"""

from copy import deepcopy
from dataclasses import asdict

from .upgrade import normalize_upgrade_title
from .hif_visual import locate_hif_selected_detail_title
from .hif_skill_support import (
    _text,
    _qualified,
    _panel_bounds,
    _verify_variant,
    _verify_selection,
    _read_detail_title,
)
from .high_score.contracts import ActionKind, _jsonable

SOURCE = 'hif_single_plus_estimate_v1'


def normalize_hif_title(value):
    return _text(value).translate(str.maketrans({'橫': '横', '顏': '顔'}))


def retained_title_row_valid(row, bundle, slot):
    """Retained identity is not strong OCR and cannot establish a new family."""
    try:
        proof = row['retained_identity']
        ref = bundle.state.deck.hand[slot]
        definition = bundle.kernel.cards[ref.card_id]
        canonical, ordinary = _verify_variant(bundle, ref, definition)
        keys = [key for key in bundle.inventory_keys if key[0] == ref.instance_id]
        suffix = proof['suffix_components']
        from .opening_support import _cross

        return bool(type(slot) is int and slot >= 0 and len(keys) == 1
            and proof['source'] == 'retained_complete_hand_selected_title'
            and tuple(proof['native_client_size']) in ((540,960),(541,961),(541,962),(542,963))
            and _jsonable(proof['binding']) == _jsonable(dict(run_id=bundle.run_id, revision=bundle.revision,
                slot=slot, card_instance_id=ref.instance_id, canonical_card_id=canonical,
                hand_instance_ids=tuple(card.instance_id for card in bundle.state.deck.hand),
                inventory_key=keys[0]))
            and ref.upgraded is True and ordinary.endswith('+') and not ordinary.endswith('++')
            and normalize_hif_title(row['text']) == ordinary
            and .85 <= row['confidence'] < .95 and row.get('text_accepted') is False
            and _qualified([dict(row, confidence=1)], proof['title_roi'])
            and len(suffix) == 1 and _cross(suffix[0],7,19)
            and min(suffix[0]['row_coverage'],suffix[0]['column_coverage']) >= .95)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return False


def _retain_selected_title(context, image, roi, rows):
    """Keep a known ordinary-plus identity with complete current text/suffix.

    Final fresh family, selection, payment and input-age checks remain separate.
    The weak text is never used to rescue an uncertain visual family.
    """
    import cv2

    from .hif_bound_title import binding
    from .opening_support import _cross, _components

    size = tuple(getattr(context, 'gkh_hif_client_size', ()))
    bundle = getattr(context, 'gkh_hif_opening_state', None)
    if (size not in ((540,960),(541,961),(541,962),(542,963)) or bundle is None
            or len(rows) != 1 or not .85 <= rows[0]['confidence'] < .95
            or not _qualified([dict(rows[0],confidence=1)],roi)):
        return rows
    x,y,w,h = roi
    mask = image[y:y+h,x:x+w,:3].min(axis=2) < 200
    ix,iy,iw,ih = cv2.boundingRect(mask.astype('uint8'))
    parts = sorted(_components(mask,(x,y)),key=lambda part:part['box'][0])
    if (ix < 8 or ix+iw > w-8 or iy < 2 or iy+ih > h-2 or len(parts) < 2
            or not _cross(parts[-1],7,19) or _cross(parts[-2],7,19)
            or not 2 <= parts[-1]['box'][0]-max(p['box'][0]+p['box'][2] for p in parts[:-1]) <= 24):
        return rows
    matches = []
    for slot in range(len(bundle.state.deck.hand)):
        bound = binding(context,slot)
        if bound is None:
            continue
        candidate = dict(rows[0], text_accepted=False, acquisition_mode='retained_complete_hand_selected_title',
            retained_identity=dict(source='retained_complete_hand_selected_title',binding=bound,
                native_client_size=list(size),title_roi=list(roi),suffix_components=[parts[-1]]))
        if retained_title_row_valid(candidate,bundle,slot):
            matches.append(candidate)
    return matches if len(matches) == 1 else rows


def read_selected_title_frame(context, image, *, frame_id):
    roi = locate_hif_selected_detail_title(image)
    if roi is None:
        raise ValueError('selected complete title is not unique')
    panel = _panel_bounds(image, roi)
    rows, raw, excluded = _read_detail_title(context, image, roi, panel)
    rows = _retain_selected_title(context,image,roi,rows)
    return dict(frame_id=frame_id, panel_box=list(panel), title_roi=list(roi),
                title_rows=rows, raw_title_rows=raw, excluded_title_pictogram_boxes=excluded)


def verify_selected_title_observations(observations, *, bundle, selected_slot,
                                       selection_evidence):
    result = dict(source=SOURCE, accepted=False, effects_verified=False,
        submission_authorized=False, model_updated=False, single_plus_estimate=False,
        requires_exact_support=False, submit_count=0, run_id=bundle.run_id,
        revision=bundle.revision, bound_state=_jsonable(asdict(bundle.state)),
        selected_slot=selected_slot, observations=deepcopy(list(observations)),
        selection_evidence=deepcopy(selection_evidence))
    try:
        if type(selected_slot) is not int or not 0 <= selected_slot < len(bundle.state.deck.hand):
            raise ValueError('selected ordinary target slot is invalid')
        ref = bundle.state.deck.hand[selected_slot]
        definition = bundle.kernel.cards[ref.card_id]
        canonical, ordinary = _verify_variant(bundle, ref, definition)
        keys = [key for key in bundle.inventory_keys if key[0] == ref.instance_id]
        if len(keys) != 1 or keys[0][1] != canonical:
            raise ValueError('selected card lacks this run permanent inventory binding')
        if '@' not in ref.card_id and keys[0][2]:
            raise ValueError('selected canonical card omits its permanent customization')
        result.update(card_instance_id=ref.instance_id, definition_id=ref.card_id,
            card_id=canonical, permanent_inventory_key=_jsonable(keys[0]))
        frame_ids = [row['frame_id'] for row in observations]
        _verify_selection(frame_ids, selected_slot, selection_evidence)
        titles = []
        for row in observations:
            panel, roi = row['panel_box'], row['title_roi']
            if (len(panel) != 4 or any(type(value) is not int for value in panel)
                    or panel[2] <= 84 or panel[3] <= 90
                    or list(roi) != [panel[0]+16, panel[1]+6, panel[2]-84, 44]
                    or len(row['title_rows']) != 1
                    or not (_qualified(row['title_rows'], roi)
                        or (row['title_rows'][0].get('retained_identity',{}).get('title_roi') == list(roi)
                            and retained_title_row_valid(row['title_rows'][0],bundle,selected_slot)))):
                raise ValueError('selected complete title is unresolved')
            titles.append(normalize_hif_title(row['title_rows'][0]['text']))
        if len(set(titles)) != 1:
            raise ValueError('selected complete title changes across frames')
        title = titles[0]
        # Preserve the established base-to-plus recovery path. D-047 does not
        # turn an unupgraded permanent card into a different permanent identity.
        if (canonical == ref.card_id and canonical in ('116', '288')
                and not ordinary.endswith('+') and title == ordinary+'+'):
            result.update(requires_exact_support=True, reason='base_card_support_requires_existing_binding')
            return result
        ordinary_projection = (canonical == ref.card_id and not ref.upgraded
                               and not ordinary.endswith('+') and title == ordinary+'+')
        if ordinary_projection:
            from .high_score.hif_support_planning import ordinary_plus_definition

            upgraded = ordinary_plus_definition(bundle.kernel.cards, definition)
            if _text(upgraded.name) != title:
                raise ValueError('selected title differs from the catalog upgrade')
            result.update(ordinary_plus_projection=True, projected_card_id=definition.upgrade_card_id)
        elif normalize_upgrade_title(title) != ordinary:
            raise ValueError('selected title differs from ordinary family or enhancement')
        count = len(title)-len(title.rstrip('+'))
        retained = any(row['title_rows'][0].get('retained_identity') for row in observations)
        result.update(accepted=True, frame_ids=frame_ids, raw_title=observations[0]['title_rows'][0]['text'],
            plus_count=count, single_plus_estimate=count > 1 or ordinary_projection,
            reason=('selected_current_ordinary_plus_projection' if ordinary_projection else
                    'd047_ordinary_plus_estimate' if count > 1 else
                    'retained_ordinary_title_and_visible_suffix' if retained else 'selected_ordinary_title_unanimous'))
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        result['reason'] = str(error)
    return result


def observe_selected_single_plus(context, frames, *, bundle, selected_slot,
                                 selection_evidence, frame_identifier):
    ids = [frame_identifier(frame) for frame in frames]
    _verify_selection(ids, selected_slot, selection_evidence)
    observations = [read_selected_title_frame(context, frame, frame_id=frame_id)
                    for frame, frame_id in zip(frames, ids)]
    return verify_selected_title_observations(observations, bundle=bundle,
        selected_slot=selected_slot, selection_evidence=selection_evidence)


def observe_selected_printed_cost_conflict(context, frames, *, bundle, action,
                                         receipt, frame_identifier):
    """A current extra-plus price contradiction requests existing exact reading.

    The header is read from the already verified complete selected panel. It
    never supplies a new definition or input authority; missing or ambiguous
    prices leave the ordinary estimate and final resource gates unchanged.
    """
    from .hif_skill_support import _cost_match, _read_printed_cost
    from .high_score.kernel import CardCostKind

    result = dict(requires_exact_support=False, observations=[],
                  reason='selected_printed_cost_not_applicable')
    if (receipt.get('accepted') is not True or type(receipt.get('plus_count')) is not int
            or receipt['plus_count'] <= 1):
        return result
    checked = verify_selected_single_plus_estimate(receipt, bundle=bundle,
                                                  action=action, require_submitted=False)
    ids = [frame_identifier(frame) for frame in frames]
    if ids != checked['frame_ids']:
        raise ValueError('selected printed cost frames differ from complete title')
    ref = bundle.state.deck.hand[action.source_slot]
    definition = bundle.kernel.card_definition(bundle.state, ref)
    if definition.cost_kind not in (CardCostKind.NORMAL, CardCostKind.STAMINA):
        return result
    result.update(frame_ids=ids, selected_slot=action.source_slot,
                  modeled_printed_cost=definition.stamina_cost)
    costs = []
    for frame, observation in zip(frames, checked['observations'], strict=True):
        x, y, w, _ = observation['panel_box']
        roi = (x+w-68, y+4, 62, 50)
        rows, original, source = _read_printed_cost(context, frame, roi)
        match = _cost_match(rows, roi)
        cost = int(match[1]) if match is not None else None
        costs.append(cost)
        result['observations'].append(dict(frame_id=observation['frame_id'],
            printed_cost=cost, roi=list(roi), rows=rows, original_rows=original, source=source))
    if len(costs) != 3 or any(value is None for value in costs) or len(set(costs)) != 1:
        result['reason'] = 'selected_printed_cost_unresolved'
        return result
    mismatch = costs[0] != definition.stamina_cost
    result.update(requires_exact_support=mismatch, observed_printed_cost=costs[0],
        reason='selected_printed_cost_differs_from_estimate' if mismatch
        else 'selected_printed_cost_matches_estimate')
    return result


def selected_ordinary_plus_visible_upgrade(bundle, policy, certificate, *, slot,
                                          frame_ids, detail, selection_evidence, identity_evidence):
    """Explain a fresh visible plus without upgrading its permanent certificate."""
    try:
        if (bundle is None or policy.bundle != bundle or type(slot) is not int
                or certificate.get('slot') != slot or certificate.get('upgraded') is not False):
            return None
        action = policy.response.action if policy.response else None
        receipt = verify_selected_single_plus_estimate(policy.selected_single_plus_estimate,
            bundle=bundle, action=action, require_submitted=False)
        if (receipt.get('ordinary_plus_projection') is not True or receipt['selected_slot'] != slot
                or certificate.get('card_id') != receipt['card_id']
                or receipt['card_id'] not in certificate.get('family', ())
                or receipt['projected_card_id'] not in certificate.get('family', ())):
            return None
        _verify_selection(frame_ids, slot, selection_evidence)
        observations = detail.get('observations', ()) if isinstance(detail, dict) else ()
        if (not isinstance(detail, dict) or detail.get('accepted') is not True
                or detail.get('card_id') != receipt['projected_card_id']
                or [row.get('frame_id') for row in observations] != frame_ids
                or any(row.get('card_id') != receipt['projected_card_id'] or row.get('roi') is None
                       for row in observations)
                or identity_evidence.get('slot') != slot
                or identity_evidence.get('accepted') is not True
                or identity_evidence.get('standard_accepted') is not True
                or identity_evidence.get('fresh_family') != list(certificate['family'])
                or identity_evidence.get('expected_family') != list(certificate['family'])):
            return None
        for row in observations:
            roi = row['roi']
            titles = [dict(text=entry['text'], box=entry['box'], confidence=entry['score'])
                      for entry in row.get('ocr', ())]
            if (len(titles) != 1 or not _qualified(titles, (0, 0, roi[2], roi[3]))
                    or _text(titles[0]['text']) != _text(receipt['raw_title'])):
                return None
        return dict(source='bound_selected_ordinary_plus_visible_upgrade', selected_slot=slot,
            card_instance_id=receipt['card_instance_id'], permanent_card_id=receipt['card_id'],
            visible_card_id=receipt['projected_card_id'], frame_ids=list(frame_ids),
            effects_verified=False, submission_authorized=False)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return None


def verify_selected_single_plus_estimate(receipt, *, bundle, action, require_submitted=True):
    """Rebuild title/SELECT proof; accepted flags alone authorize no reconciliation."""
    if (not isinstance(receipt, dict) or receipt.get('source') != SOURCE
            or receipt.get('run_id') != bundle.run_id or receipt.get('revision') != bundle.revision
            or receipt.get('bound_state') != _jsonable(asdict(bundle.state))
            or type(receipt.get('submit_count')) is not int
            or receipt['submit_count'] != (1 if require_submitted else 0)
            or action is None or action.kind is not ActionKind.PLAY_CARD
            or type(action.source_slot) is not int
            or action.source_slot != receipt.get('selected_slot')
            or action.card_instance_id != receipt.get('card_instance_id')):
        raise ValueError('single-plus estimate is not bound to this committed action and state')
    verified = verify_selected_title_observations(receipt.get('observations', ()), bundle=bundle,
        selected_slot=action.source_slot, selection_evidence=receipt.get('selection_evidence', {}))
    if not verified['accepted'] or not verified['single_plus_estimate']:
        raise ValueError('single-plus estimate lacks complete projected title and SELECT evidence')
    for key in ('definition_id', 'card_instance_id', 'permanent_inventory_key'):
        if receipt.get(key) != verified[key]:
            raise ValueError('single-plus estimate differs from current permanent inventory')
    verified['submit_count'] = receipt['submit_count']
    return verified
