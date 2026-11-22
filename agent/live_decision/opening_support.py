"""Bind the observed opening-only blue support mark without upgrading the deck."""
import math
from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np

from .upgrade import detect_live_upgrade_marker
from .inventory import match_inventory_hand
from .hif_visual import _hif_gray_title_plates, _hif_unselected_title_plates
from .opening_observer import opening_hand_titles

SOURCE = 'hif_opening_temporary_support_v1'
TITLE = 'ひと呼吸+'


def _components(mask, origin):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    result = []
    for index in range(1, count):
        x, y, width, height, area = map(int, stats[index])
        component = labels[y:y+height, x:x+width] == index
        rows, columns = component.sum(axis=1), component.sum(axis=0)
        result.append(dict(box=[x+origin[0], y+origin[1], width, height], area=area,
            row_coverage=float(rows.max()/width), column_coverage=float(columns.max()/height),
            row_center=int(rows.argmax()), column_center=int(columns.argmax())))
    return result


def _cross(row, minimum, maximum):
    _, _, width, height = row['box']
    return (minimum <= width <= maximum and minimum <= height <= maximum
        and .65 <= width/height <= 1.45 and .15 <= row['area']/(width*height) <= .7
        and .8 <= row['row_coverage'] <= 1 and .8 <= row['column_coverage'] <= 1
        and abs(row['row_center']-(height-1)/2) <= 3.5
        and abs(row['column_center']-(width-1)/2) <= 3.5)


def read_opening_support_frame(image, *, frame_id, slot, title_row):
    """Observe the actual 997 three-card layout; no effect amount is inferred."""
    if tuple(image.shape) != (1280, 720, 3) or type(slot) is not int or slot not in (0, 1, 2):
        raise ValueError('opening support requires the calibrated three-card row')
    anchor = (48, 265, 482)[slot]
    return _read_support_frame(image, frame_id=frame_id, slot=slot, title_row=title_row, anchor=anchor)


def _read_support_frame(image, *, frame_id, slot, title_row, anchor, title_plate=None,
                        title_normalization=None):
    available = (title_plate,) if title_plate is not None else _hif_unselected_title_plates(image)
    plates = [box for box in available
              if abs(box[0]-anchor) <= 5 and 180 <= box[2] <= 190]
    if len(plates) != 1:
        raise ValueError('opening support title plate is incomplete')
    title = dict(text=title_row.text, score=title_row.score, box=list(title_row.box))
    complete_title_observation = None
    if getattr(title_row, 'acquisition_mode', None) == 'native_complete_title_plate':
        # The complete-plate OCR box includes blank padding. Its retained page
        # detection supplies the text bounds; both independently read values
        # must agree before measuring the blue suffix against those bounds.
        detection = getattr(title_row, 'source_plate_observation', None)
        if detection is not None and (detection.get('roi') != list(plates[0])
                or detection.get('acquisition_mode') != 'native_complete_title_detection'):
            raise ValueError('complete support title bounds belong to another plate')
        detection = detection or getattr(title_row, 'source_page_observation', {})
        box = detection.get('box', ())
        px, py, pw, ph = plates[0]
        if (getattr(title_row, 'complete_title_plate', False) is not True
                or tuple(title_row.box) != tuple(plates[0])
                or detection.get('text') != title_row.text
                or type(detection.get('score')) not in (int, float)
                or not .95 <= detection['score'] <= 1
                or len(box) != 4 or any(type(value) is not int for value in box)
                or not (px <= box[0] < box[0]+box[2] <= px+pw
                        and py <= box[1] < box[1]+box[3] <= py+ph)):
            raise ValueError('complete support title and retained text bounds disagree')
        complete_title_observation = dict(title,
            source_page_observation=deepcopy(getattr(title_row, 'source_page_observation', {})),
            source_plate_observation=deepcopy(getattr(title_row, 'source_plate_observation', None)))
        title.update(box=list(box), score=min(title_row.score, detection['score']))
    roi = [anchor, 1090, 190, 44]
    crop = image[1090:1134, anchor:anchor+190]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (85, 100, 100), (112, 255, 255))
    crosses = [row for row in _components(blue, (anchor, 1090)) if _cross(row, 7, 13)]
    if len(crosses) != 1:
        raise ValueError('opening support blue suffix is missing or ambiguous')
    ordinary = detect_live_upgrade_marker(image, slot, box=tuple(roi))
    raw_marker = None
    if title_normalization is not None:
        proof = title_normalization
        if (proof.get('slot') != slot or proof.get('gain') != 2
                or tuple(proof.get('title_plate',())) != tuple(plates[0])
                or proof.get('gray_level',0)*2 != proof.get('white_level')
                or proof.get('white_level',0) < 235):
            raise ValueError('opening support title intensity proof is incomplete')
        # The independent blue suffix and badge above/below still use raw
        # pixels. Only the ordinary dark-ink marker needs its proven white
        # title level restored; otherwise the entire gray plate is ink.
        raw_marker = ordinary.to_dict()
        corrected = image.copy()
        px,py,pw,ph = plates[0]
        corrected[py:py+ph,px:px+pw] = np.minimum(
            image[py:py+ph,px:px+pw].astype(np.uint16)*2,255).astype(np.uint8)
        ordinary = detect_live_upgrade_marker(corrected,slot,box=tuple(roi))
    marker = ordinary.to_dict()
    marker['box'] = list(marker['box'])
    badge_roi = [anchor+118, 920, 72, 95]
    x, y, width, height = badge_roi
    hsv = cv2.cvtColor(image[y:y+height, x:x+width], cv2.COLOR_BGR2HSV)
    warm = (((hsv[:, :, 0] <= 40) | (hsv[:, :, 0] >= 165))
            & (hsv[:, :, 1] >= 140) & (hsv[:, :, 2] >= 190))
    components = _components(warm, (x, y))
    borders = [row for row in components if 45 <= row['box'][2] <= 65
               and 45 <= row['box'][3] <= 65
               and .08 <= row['area']/(row['box'][2]*row['box'][3]) <= .3]
    pluses = [row for row in components if _cross(row, 20, 30)]
    if len(borders) != 1 or len(pluses) != 1:
        raise ValueError('opening support portrait badge is incomplete or ambiguous')
    return dict(frame_id=frame_id, slot=slot, title=title, title_plate=list(plates[0]),
        suffix_roi=roi, suffix=crosses[0], suffix_color='blue',
        marker=marker, badge_roi=badge_roi,
        badge_border=borders[0], badge_plus=pluses[0],
        **({'complete_title_observation': complete_title_observation}
           if complete_title_observation is not None else {}),
        **(dict(raw_marker=raw_marker,title_normalization=deepcopy(title_normalization))
           if raw_marker is not None else {}))


def _verify_frame(row, slot, *, anchor=None, title_text=TITLE, suffix_right_margin=4):
    if anchor is None:
        anchor = (48, 265, 482)[slot]
    title, plate, suffix = row['title'], row['title_plate'], row['suffix']
    x, y, width, height = title['box']
    if (row['slot'] != slot or title['text'] != title_text
            or type(title['score']) not in (int, float) or not math.isfinite(title['score'])
            or not .95 <= title['score'] <= 1
            or not (anchor <= x < x+width <= anchor+190 and 1090 <= y <= 1120
                    and 14 <= height <= 32)
            or len(plate) != 4 or abs(plate[0]-anchor) > 5 or plate[1] != 1100
            or not (180 <= plate[2] <= 190 and 30 <= plate[3] <= 34)
            or row['suffix_roi'] != [anchor, 1090, 190, 44]
            or row['suffix_color'] != 'blue' or not _cross(suffix, 7, 13)
            or not (x <= suffix['box'][0] and y <= suffix['box'][1]
                    and suffix['box'][1]+suffix['box'][3] <= y+height
                    and abs(suffix['box'][0]+suffix['box'][2]-(x+width)) <= suffix_right_margin)
            or row['marker']['upgraded'] is not True or row['marker']['marker_count'] != 1
            or row['marker']['confidence'] < .8
            or list(row['marker']['box']) != row['suffix_roi']):
        raise ValueError('opening support complete title or blue suffix is unresolved')
    border, plus = row['badge_border'], row['badge_plus']
    bx, by, bw, bh = border['box']
    px, py, pw, ph = plus['box']
    if (row['badge_roi'] != [anchor+118, 920, 72, 95]
            or not (anchor+118 <= bx <= anchor+145 and 925 <= by <= 955
                    and 45 <= bw <= 65 and 45 <= bh <= 65
                    and .08 <= border['area']/(bw*bh) <= .3)
            or not _cross(plus, 20, 30)
            or not (bx+bw*.45 <= px and by+bh*.4 <= py
                    and px+pw <= bx+bw+2 and py+ph <= by+bh+2)):
        raise ValueError('opening support portrait badge does not match its card')


def _read_occluded_bound_support_frame(image, *, frame_id, slot, title_row, candidates,
                                      selected_slot=None):
    """1187: complete blue title, with the portrait badge behind its neighbour.

    The visible title retains an already bound permanent identity only. The
    occluded portrait plus is not observed and cannot supply effect evidence.
    """
    gray_plates = _hif_gray_title_plates(image)
    plates = _hif_unselected_title_plates(image, gray_plates=gray_plates)
    if selected_slot is not None:
        from .hif_visual import hif_selected_row_proposal

        proposal = hif_selected_row_proposal(image, candidates, hand_count=4)
        # A different raised card does not move this title or its right-hand
        # occluder. Prove the complete row again, including the selected body.
        if (type(selected_slot) is not int or not 0 <= selected_slot < 4
                or selected_slot in (slot, slot+1) or len(plates) != 3
                or proposal is None or proposal[1] != selected_slot
                or tuple(c.box for c in proposal[0]) != tuple(c.box for c in candidates)):
            raise ValueError('bound support selected neighbour geometry changed')
        plates = (*plates[:selected_slot], None, *plates[selected_slot:])
    if len(plates) != 4 or len(candidates) != 4 or slot not in (0, 1, 2):
        raise ValueError('bound support lacks the complete unselected four-card row')
    for index, (plate, candidate) in enumerate(zip(plates, candidates, strict=True)):
        if index == selected_slot:
            continue
        anchor = plate[0]-2
        expected = 18 + index*494/3
        gray_neighbour = (index != slot and plate in gray_plates
            and (candidate.detector_label == 'hif_gray_observation'
                 or selected_slot is not None and candidate.detector_label == 'useless')
            and candidate.clickable is False)
        # A grey neighbour's independently bordered title may end one pixel
        # lower or be narrower than a white title. Use the same proven body
        # as the row normalizer; it cannot supply the target's blue suffix.
        top = plate[1]+plate[3]+2-250 if gray_neighbour else 884
        if gray_neighbour and selected_slot is not None:
            # The selected-row normalizer has already proved this grey body
            # at its canonical anchor; the title's one-pixel inset may differ.
            anchor, top = round(expected), 884
        if (abs(anchor-expected) > 5 or candidate.slot != index
                or tuple(candidate.box) != (anchor, top, 190, 250)
                or plate[1] != 1100 or not 30 <= plate[3] <= 34
                or not gray_neighbour and abs(plate[2]-(162 if index < 3 else 186)) > 3):
            raise ValueError('bound support four-card geometry changed')
    plate = plates[slot]
    anchor, boundary = plate[0]-2, plates[slot+1][0]-2
    # The normal 190px card is covered on the right. The next card must be
    # precisely the owner of that missing strip, not a selected/raised card.
    if not 162 <= boundary-anchor <= 167 or abs(plate[0]+plate[2]-boundary) > 3:
        raise ValueError('bound support portrait occlusion is not uniquely owned')
    title = dict(text=title_row.text, score=title_row.score, box=list(title_row.box))
    x, y, w, h = title['box']
    if (type(title['score']) not in (int, float) or not math.isfinite(title['score'])
            or not .95 <= title['score'] <= 1 or not 1090 <= y <= 1120
            or not 14 <= h <= 32 or not (plate[0] <= x < x+w <= boundary)):
        raise ValueError('bound support complete title is outside its visible plate')
    roi = [anchor, 1090, boundary-anchor, 44]
    crop = image[1090:1134, anchor:boundary]
    blue = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), (85, 100, 100), (112, 255, 255))
    crosses = [row for row in _components(blue, (anchor, 1090)) if _cross(row, 7, 13)]
    if len(crosses) != 1:
        raise ValueError('bound support blue suffix is missing or ambiguous')
    suffix = crosses[0]
    sx, sy, sw, sh = suffix['box']
    marker = detect_live_upgrade_marker(image, slot, box=tuple(roi)).to_dict()
    if (not (x <= sx and y <= sy and sy+sh <= y+h and abs(sx+sw-x-w) <= 6)
            or marker['upgraded'] is not True or marker['marker_count'] != 1
            or marker['confidence'] < .8):
        raise ValueError('bound support exact blue title suffix is unresolved')
    return dict(frame_id=frame_id, slot=slot, title=title, title_plate=list(plate),
        suffix_roi=roi, suffix=suffix, suffix_color='blue', marker=marker,
        badge_occluded=True, badge_occluder_slot=slot+1, badge_occlusion_x=boundary,
        unselected_title_plates=[list(value) if value is not None else None for value in plates],
        **(dict(selected_slot=selected_slot) if selected_slot is not None else {}))


def _complete_support_title_bounds(context, image, title):
    """Re-detect inside the same full plate when page detection crosses its edge."""
    from .inventory import _inventory_ocr

    if getattr(title, 'source_plate_observation', None) is not None:
        # The complete-plate reader already detected text inside this plate.
        # _read_support_frame independently validates its text, ROI and bounds.
        return title
    if getattr(title, 'acquisition_mode', None) != 'native_complete_title_plate':
        # The legacy 33-36px title reader preserves its page box after an
        # independent crop read. Establish the actual plate before detecting
        # suffix bounds, using the same complete text and confidence gates.
        x, y, w, h = title.box
        if (getattr(title, 'complete_title_plate', False) is not True
                or not .95 <= title.score <= 1 or not 32 < h <= 36):
            return title
        plates = [p for p in _hif_unselected_title_plates(image)
            if 180 <= p[2] <= 190 and 30 <= p[3] <= 34
            and p[0]-2 <= x < x+w <= p[0]+p[2]+2
            and p[1]-4 <= y < y+h <= p[1]+p[3]+2]
        if len(plates) != 1:
            return title
        px, py, pw, ph = plates[0]
        rows = _inventory_ocr(context, image[py:py+ph, px:px+pw].copy(),
            'HifSupportCompleteTitlePlate', only_rec=True)
        if len(rows) != 1 or rows[0].text != title.text or not .95 <= rows[0].score <= 1:
            return title
        title = SimpleNamespace(text=rows[0].text, score=rows[0].score, box=plates[0],
            complete_title_plate=True, acquisition_mode='native_complete_title_plate',
            source_page_observation=dict(text=title.text, score=title.score, box=list(title.box)))
    px, py, pw, ph = title.box
    source = title.source_page_observation
    x, y, w, h = source['box']
    if px <= x < x+w <= px+pw and py <= y < y+h <= py+ph:
        return title
    if (source['text'] != title.text or not .95 <= source['score'] <= 1
            or not (px-2 <= x < x+w <= px+pw+2 and py-4 <= y < y+h <= py+ph+2)):
        return title
    rows = _inventory_ocr(context, image[py:py+ph, px:px+pw].copy(),
        'HifSupportCompleteTitleBounds', only_rec=False)
    if len(rows) != 1 or rows[0].text != title.text or not .95 <= rows[0].score <= 1:
        return title
    row = rows[0]
    x, y, w, h = row.box
    if not (0 <= x < x+w <= pw and 0 <= y < y+h <= ph):
        return title
    return SimpleNamespace(**vars(title), source_plate_observation=dict(
        text=row.text, score=row.score, box=[px+x, py+y, w, h], roi=list(title.box),
        acquisition_mode='native_complete_title_detection'))


def observe_bound_rightmost_support(context, frames, *, bundle, slot, frame_identifier,
                                    candidates=(), selected_slot=None):
    """1082: preserve a complete permanent identity below a fresh blue suffix.

    The full-badge path covers calibrated one-to-three-card rows and the last card
    of five. Native 1187's four-card overlap occludes the badge.
    It grants no selected effect, new permanent level, or input permission.
    """
    from .opening_observer import valid_opening_frame_ids, read_opening_identity_titles

    hand_count = len(bundle.state.deck.hand)
    short_anchors = {1: (265,), 2: (157,374), 3: (48,265,482)}
    full_short = hand_count in short_anchors and 0 <= slot < hand_count
    occluded_four = len(bundle.state.deck.hand) == 4 and slot in (0, 1, 2)
    full_last = len(bundle.state.deck.hand) in (4, 5) and slot == len(bundle.state.deck.hand)-1
    if not (full_short or full_last or occluded_four) or not frames:
        raise ValueError('bound support requires the complete rightmost five-card title')
    if full_short and (len(candidates) != hand_count or selected_slot == slot
            or selected_slot is not None and selected_slot not in range(hand_count)
            or any(candidate.slot != index for index, candidate in enumerate(candidates))
            or abs(candidates[slot].box[0]-short_anchors[hand_count][slot]) > 8
            or candidates[slot].box[2:] != (190,250)):
        raise ValueError('bound support complete three-card geometry changed')
    fresh_short_geometry = full_short and tuple(candidates[slot].box) != (short_anchors[hand_count][slot],884,190,250)
    ref = bundle.state.deck.hand[slot]
    definition = bundle.kernel.cards[ref.card_id]
    visible = bundle.kernel.cards.get(definition.upgrade_card_id)
    if (ref.upgraded or (ref.instance_id, ref.card_id, ()) not in bundle.inventory_keys
            or visible is None or visible.base_card_id != definition.base_card_id
            or visible.name != definition.name+'+'
            or sum(row.card_id == ref.card_id for row in bundle.state.deck.hand) != 1):
        raise ValueError('bound support lacks an unambiguous permanent base card')
    observations = []
    for frame in frames:
        if tuple(frame.shape) != (1280, 720, 3):
            raise ValueError('bound support frame geometry changed')
        anchor = (short_anchors[hand_count][slot] if full_short
                  else 18 + slot*494/3 if occluded_four else 512)
        full_plate = title_normalization = None
        if fresh_short_geometry:
            from .hif_visual import hif_selected_row_proposal, _hif_disabled_title_plates, hif_unselected_row_candidates

            if selected_slot is None:
                complete = hif_unselected_row_candidates(frame.shape,candidates,image=frame)
            else:
                proposal = hif_selected_row_proposal(frame,candidates,hand_count=hand_count)
                complete = proposal[0] if proposal and proposal[1] == selected_slot else ()
            if (not complete or tuple(c.box for c in complete) != tuple(c.box for c in candidates)):
                raise ValueError('bound support complete three-card geometry changed')
            plates = _hif_unselected_title_plates(frame)
            plates = (*plates,*_hif_disabled_title_plates(frame,candidates,plates))
            owned = [p for p in plates if (p[0]-2,p[1]+p[3]+2-250,190,250) == candidates[slot].box]
            if len(owned) != 1:
                raise ValueError('bound support complete three-card geometry changed')
            full_plate,anchor = owned[0],candidates[slot].box[0]
            from .partial_card import build_live_card_views, normalize_hif_disabled_art

            views = build_live_card_views(frame.shape,candidates)
            _corrected,proofs = normalize_hif_disabled_art(frame,candidates,views,selected_slot=selected_slot)
            title_normalization = next((proof for proof in proofs if proof['slot'] == slot),None)
        titles = [row for row in read_opening_identity_titles(context, frame)
                  if row.text == visible.name and row.box[0] >= anchor and 1090 <= row.box[1] <= 1120]
        if len(titles) != 1:
            raise ValueError('bound support full title is not unique')
        if occluded_four:
            row = _read_occluded_bound_support_frame(frame, frame_id=frame_identifier(frame),
                slot=slot, title_row=titles[0], candidates=candidates, selected_slot=selected_slot)
        else:
            title = _complete_support_title_bounds(context, frame, titles[0])
            row = _read_support_frame(frame, frame_id=frame_identifier(frame),
                slot=slot, title_row=title, anchor=anchor,
                **(dict(title_plate=full_plate,title_normalization=title_normalization)
                   if full_plate is not None else {}))
            # 1082's complete title detector includes 4-5 pixels after the blue ink.
            _verify_frame(row, slot, anchor=anchor, title_text=visible.name, suffix_right_margin=6)
        observations.append(row)
    if not valid_opening_frame_ids([row['frame_id'] for row in observations]):
        raise ValueError('bound support lacks an observed frame')
    return dict(source='bound_rightmost_blue_support_identity', run_id=bundle.run_id,
        revision=bundle.revision, observed_turn=bundle.state.turn, slot=slot,
        card_instance_id=ref.instance_id, card_id=ref.card_id, visible_card_id=visible.card_id,
        observations=observations, effects_verified=False, submission_authorized=False)


def verify_opening_skill_support_receipt(receipt, *, run_id, revision,
                                         card_instance_id, card_id, slot):
    """Replay raw observations; derived permission flags grant no authority."""
    try:
        if (type(slot) is not int or slot not in (0, 1, 2) or card_id != '288'
                or receipt['source'] != SOURCE or receipt['run_id'] != run_id
                or receipt['revision'] != revision or receipt['card_instance_id'] != card_instance_id
                or receipt['card_id'] != card_id or receipt['visible_card_id'] != '289'
                or receipt['slot'] != slot or receipt['observed_turn'] != 1
                or receipt['effects_verified'] is not False
                or receipt['submission_authorized'] is not False):
            raise ValueError('opening support is outside its verified instance and epoch')
        observations = receipt['observations']
        ids = [row['frame_id'] for row in observations]
        from .opening_observer import valid_opening_frame_ids

        if not valid_opening_frame_ids(ids) or list(receipt['frame_ids']) != ids:
            raise ValueError('opening support lacks its observed frames')
        for row in observations:
            _verify_frame(row, slot)
        return deepcopy(receipt)
    except (KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        raise ValueError('opening support observation is incomplete') from error


def observe_opening_skill_support(frames, title_frames, *, run_id, revision,
                                  core, inventory_rows):
    """Only the known base 288 can retain identity beneath its observed blue +."""
    hand = [row for row in inventory_rows if row['zone'] == '手札']
    ids = match_inventory_hand(core['families'], tuple(row['card_id'] for row in hand))
    titles = tuple(opening_hand_titles(rows) for rows in title_frames)
    if not frames or len(titles) != len(frames) or any(title != titles[0] for title in titles[1:]):
        raise ValueError('opening support full titles do not agree')
    receipts = []
    for slot, (card_id, title) in enumerate(zip(ids, titles[0], strict=True)):
        if card_id != '288' or title != TITLE:
            continue
        physical = [row for row in hand if row['card_id'] == card_id]
        if (len(physical) != 1 or physical[0]['customizations']
                or physical[0]['source'] != 'observed_badge_absence'):
            raise ValueError('opening support base inventory identity is ambiguous')
        instance = f'{run_id}:手札:{physical[0]["index"]}'
        observations = []
        for frame, rows, frame_id in zip(frames, title_frames, core['frame_ids'], strict=True):
            matching = [row for row in rows if row.text == TITLE
                        and slot*240 <= row.box[0]+row.box[2]/2 < (slot+1)*240
                        and 1090 <= row.box[1] <= 1120]
            if len(matching) != 1:
                raise ValueError('opening support full title is ambiguous')
            observations.append(read_opening_support_frame(frame, frame_id=frame_id,
                slot=slot, title_row=matching[0]))
        receipt = dict(source=SOURCE, run_id=run_id, revision=revision,
            card_instance_id=instance, card_id=card_id, visible_card_id='289', slot=slot,
            observed_turn=1, frame_ids=list(core['frame_ids']), observations=observations,
            effects_verified=False, submission_authorized=False)
        receipts.append(verify_opening_skill_support_receipt(receipt, run_id=run_id,
            revision=revision, card_instance_id=instance, card_id=card_id, slot=slot))
    return tuple(receipts)
