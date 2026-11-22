"""Read the observed, single-page HIF retrieval grid without sending input.

The 720 x 1280 layout is calibrated from the retained 871 selection page.
Card identities come from the existing thumbnail reader, never from grid order.
"""

import math
from types import SimpleNamespace
from collections import Counter, defaultdict
from dataclasses import asdict

import cv2
import numpy as np

from .inventory import _inventory_ocr, _unique_inventory_text
from .high_score.contracts import ActionKind, DeckKnowledge
from .inventory_customization import (
    measure_customization_badge,
    retained_permanent_card_rows,
    stable_customization_badge_presence,
)

CHOICE_BOXES = tuple((x, y, 118, 118) for y in (399, 546, 693, 840, 987)
                     for x in (81, 228, 375, 521))
# In the retained 1583 frame, the list clips at this normalized row. The
# final row's lower horizontal bracket arms (y=1109..1116) are outside it.
CHOICE_GRID_CLIP_BOTTOM = 1107
# The existing independent color probes lie wholly inside the button. Use
# the first currently verified probe as the click region, not a jittering OCR
# text box. Both probes and the complete title still have to pass each read.
CHOICE_CONFIRM_PATCHES = ((305, 1145, 25, 30), (425, 1145, 25, 30))


def hif_choice_grid(frame, *, expected_count=None, target_slot=None, scroll_offset=None):
    """Read a complete visible prefix, independently of the deck's card count."""
    array = np.asarray(frame)
    if array.shape != (1280, 720, 3) or array.dtype != np.uint8:
        raise ValueError('HIF choice resolution is not calibrated')
    edges = cv2.Canny(cv2.cvtColor(array, cv2.COLOR_BGR2GRAY), 50, 150)
    present = []
    boxes = CHOICE_BOXES
    if target_slot is not None:
        if type(target_slot) is not int or type(expected_count) is not int or not 0 <= target_slot < expected_count:
            raise ValueError('HIF choice target is outside the bound list')
        if scroll_offset is None:
            if target_slot >= len(boxes):
                raise ValueError('HIF choice target is outside the bound list')
            boxes = (boxes[target_slot],)
        else:
            from .hif_choice_scroll import visible_move_cells
            visible = dict(visible_move_cells(expected_count, scroll_offset))
            if target_slot not in visible:
                raise ValueError('HIF choice target is outside the current move viewport')
            boxes = (visible[target_slot],)
    for x, y, width, height in boxes:
        # Rounded corners, plus badges and live numbers do not cover these
        # long horizontal borders. The broad search only locates the border;
        # identity still has to pass the unchanged production reader.
        # Resampling a 480x853 or 480x854 client to this normalized page can
        # split one horizontal edge over two adjacent rows. Count supported
        # columns in that same narrow edge band, not disconnected lengths.
        top = max(int(np.count_nonzero(np.any(edges[row:row+2, x+10:x+width-10], axis=0)))
                  for row in range(y-4, y+5))
        bottom = max(int(np.count_nonzero(np.any(edges[row:row+2, x+10:x+width-10], axis=0)))
                     for row in range(y+height-6, y+height+5))
        if top >= 72 and bottom < 72:
            # The retained 1755/1756 card rims split between adjacent inner
            # and outer edges two rows apart. Combine that three-row rim only
            # when both independent vertical sides also prove this cell.
            left = max(int(np.count_nonzero(np.any(edges[y+10:y+height-10, col:col+2], axis=1)))
                       for col in range(x-4, x+5))
            right = max(int(np.count_nonzero(np.any(edges[y+10:y+height-10, col:col+2], axis=1)))
                        for col in range(x+width-6, x+width+5))
            joined_bottom = max(int(np.count_nonzero(np.any(edges[row:row+3, x+10:x+width-10], axis=0)))
                                for row in range(y+height-6, y+height+4))
            if min(left, right, joined_bottom) >= 72:
                bottom = joined_bottom
        present.append(top >= 72 and bottom >= 72)
    if target_slot is not None:
        if not present[0]:
            raise ValueError('HIF choice target is not visible in its bound slot')
        # The other boxes are the existing layout, not newly observed identities.
        if scroll_offset is None:
            return CHOICE_BOXES[:expected_count]
        return tuple(visible.get(slot) for slot in range(expected_count))
    count = sum(present)
    if not 1 <= count <= len(CHOICE_BOXES) or present != [True] * count + [False] * (20-count):
        raise ValueError('HIF choice grid is incomplete or differs from the calibrated page')
    if expected_count is not None:
        if type(expected_count) is not int or count != expected_count:
            raise ValueError('HIF choice visible grid differs from the pending retrieval boundary')
        # The known complete candidate set determines extent. A background
        # hand card at the old scrollbar coordinate is not scrolling evidence.
        # The caller still binds every visible identity to that exact set.
        return CHOICE_BOXES[:count]
    bar = array[399:1081, 682:685].astype(np.int16)
    gray = ((bar.max(axis=2)-bar.min(axis=2) <= 16)
            & (bar.mean(axis=2) >= 90) & (bar.mean(axis=2) <= 200))
    if not bool(np.all(gray)):
        # The actual 1773 four-row and 2000 two-row pages have no scrollbar.
        # All identities still have to match the pending set before binding. A
        # partial solid scrollbar is not absence: retain the existing color
        # gate and detect a square of its calibrated three-pixel width.
        solid = np.all(bar == bar[:, :1], axis=(1, 2)) & np.all(gray, axis=1)
        partial = (solid[:-2] & solid[1:-1] & solid[2:]
                   & np.all(bar[:-2, 0] == bar[1:-1, 0], axis=1)
                   & np.all(bar[:-2, 0] == bar[2:, 0], axis=1))
        if count > 16 or bool(np.any(partial)):
            raise ValueError('HIF choice list extent is not completely visible')
    return CHOICE_BOXES[:count]


def _choice_selected_slot(frame, boxes):
    """Locate all four external corners, accounting for the last-row clip."""
    hsv = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2HSV)
    orange = cv2.inRange(hsv, (5, 100, 180), (35, 255, 255)) > 0
    marked = []
    for slot, (x, y, width, height) in enumerate(boxes):
        arms, visible_arms = [], []
        clipped = (x, y, width, height) in CHOICE_BOXES[16:]
        clip_edge_observed = clipped
        for side_x, side_y in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            corner_x = x if side_x < 0 else x + width
            corner_y = y if side_y < 0 else y + height
            left, right = ((corner_x-12, corner_x+24) if side_x < 0
                           else (corner_x-24, corner_x+12))
            top, bottom = ((corner_y-12, corner_y-4) if side_y < 0
                           else (corner_y+4, corner_y+12))
            horizontal = float(orange[top:bottom, left:right].mean())
            left, right = ((corner_x-12, corner_x-4) if side_x < 0
                           else (corner_x+4, corner_x+12))
            top, bottom = ((corner_y-12, corner_y+24) if side_y < 0
                           else (corner_y-24, corner_y+12))
            vertical = float(orange[top:bottom, left:right].mean())
            arms.extend((horizontal, vertical))
            if clipped and side_y > 0 and corner_y + 4 >= CHOICE_GRID_CLIP_BOTTOM:
                # Require both lower vertical arms up to the calibrated edge,
                # then their simultaneous disappearance below it. Missing
                # corners elsewhere cannot use this clipped-page exception.
                edge = CHOICE_GRID_CLIP_BOTTOM
                visible_arms.extend((float(orange[top:edge, left:right].mean()),
                                     float(orange[edge-4:edge, left:right].mean())))
                clip_edge_observed &= not bool(np.any(orange[edge:edge+8, left:right]))
            else:
                visible_arms.extend((horizontal, vertical))
        if min(arms) >= .5 or (clip_edge_observed and min(visible_arms) >= .5):
            marked.append(slot)
    if len(marked) > 1:
        raise ValueError('HIF choice selected brackets are not unique')
    return marked[0] if marked else None


def _choice_confirm_enabled(frame):
    """Observe solid enabled orange versus the translucent disabled button."""
    hsv = cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2HSV)
    states = []
    # Interior patches avoid the white check mark, title and capsule border.
    for x, y, width, height in CHOICE_CONFIRM_PATCHES:
        patch = hsv[y:y+height, x:x+width]
        hue, saturation, value = (patch[:, :, index] for index in range(3))
        enabled = (hue >= 5) & (hue <= 30) & (saturation >= 175) & (value >= 240)
        disabled = ((hue >= 5) & (hue <= 30) & (saturation >= 100)
                    & (saturation <= 200) & (value >= 160) & (value <= 210))
        states.append(True if enabled.mean() >= .95 else False if disabled.mean() >= .95 else None)
    if states[0] is None or states[0] != states[1]:
        raise ValueError('HIF choice confirmation button state is ambiguous')
    return states[0]


def _choice_selected_title_rows(context, frame):
    if getattr(context, 'gkh_hif_upstream_card_ui', False) is True:
        # fb8 _read_hif_move_card_title crops the title, then joins OCR hits left to right.
        x, y, w, h = 190, 105, 410, 40
        rows = _inventory_ocr(context, frame[y:y+h, x:x+w].copy(), 'ProduceRecognitionScore',
                              roi=(0, 0, w, h), expected=r'[^\n]+')
        if not rows:
            return ()
        text = ''.join(row.text or '' for row in sorted(rows, key=lambda row: row.box[0]))
        return (SimpleNamespace(text=text, score=min(row.score for row in rows), box=(x, y, w, h)),)
    roi = (185, 102, 440, 48)
    rows = _inventory_ocr(context, frame, 'HifChoiceSelectedTitle', roi=roi)
    if len(rows) == 1:
        row = rows[0]
        score, box = row.score, tuple(row.box)
        x, y, width, height = roi
        if (type(score) in (float, int) and math.isfinite(score) and 0 <= score < .95
                and row.text.strip() and len(box) == 4
                and all(type(value) is int for value in box)
                and min(box[2:]) > 0 and x <= box[0] and y <= box[1]
                and box[0]+box[2] <= x+width and box[1]+box[3] <= y+height):
            # One fixed enlargement of the same complete title region. The
            # retained 1761 weak read already has the full correct text; a
            # different title, partial box or another weak read is no proof.
            crop = np.asarray(frame)[y:y+height, x:x+width]
            enlarged = cv2.resize(crop, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
            reread = _inventory_ocr(context, enlarged, 'HifChoiceSelectedTitleDouble')
            if len(reread) == 1:
                fresh = reread[0]
                bounds = tuple(fresh.box)
                if (fresh.text.strip() == row.text.strip()
                        and type(fresh.score) in (float, int) and math.isfinite(fresh.score)
                        and .95 <= fresh.score <= 1 and len(bounds) == 4
                        and all(type(value) is int for value in bounds)
                        and min(bounds[2:]) > 0 and min(bounds[:2]) >= 0
                        and bounds[0]+bounds[2] <= width*2
                        and bounds[1]+bounds[3] <= height*2):
                    left, top = bounds[0]//2, bounds[1]//2
                    right = math.ceil((bounds[0]+bounds[2])/2)
                    bottom = math.ceil((bounds[1]+bounds[3])/2)
                    rows = (SimpleNamespace(text=fresh.text, score=fresh.score,
                        box=(x+left, y+top, right-left, bottom-top)),)
    return rows


def read_hif_choice_panel(context, frame, *, expected_count=None, target_slot=None, scroll_offset=None):
    """Read selected or unselected UI; this observation authorizes no input."""
    if target_slot is None:
        boxes = hif_choice_grid(frame, expected_count=expected_count)
        selected_slot = _choice_selected_slot(frame, boxes)
    else:
        geometry = {} if scroll_offset is None else dict(scroll_offset=scroll_offset)
        boxes = hif_choice_grid(frame, expected_count=expected_count, target_slot=target_slot, **geometry)
        marked = _choice_selected_slot(frame, (boxes[target_slot],))
        selected_slot = target_slot if marked is not None else None
    upstream = getattr(context, 'gkh_hif_upstream_card_ui', False) is True
    if upstream:
        # fb8 _handle_move_cards recognizes the untouched prompt. After a
        # selection _wait_for_move_enabled recognizes the actionable button.
        node = 'ProduceMoveCards'
        control = context.run_recognition(node, frame)
        enabled = bool(control and control.hit)
        if not enabled:
            node = 'ProduceRecognitionChooseMoveCards'
            control = context.run_recognition(node, frame)
            if not control or not control.hit:
                raise ValueError('HIF choice confirmation button or unselected prompt is absent')
        match = control.best_result
        anchors = [dict(node=node, box=tuple(match.box), score=float(match.score))]
        confirm_target = tuple(match.box) if enabled else None
    else:
        enabled = _choice_confirm_enabled(frame)
        confirm_target = CHOICE_CONFIRM_PATCHES[0]
    if (not enabled and selected_slot is not None) or (target_slot is None and enabled and selected_slot is None):
        raise ValueError('HIF choice selected marker and confirmation button disagree')
    if not upstream:
        anchors = []
        labels = [
            ('HifChoiceTitle', '移動', (0, 25, 300, 65)),
            ('HifChoiceConfirm', '移動', (225, 1110, 280, 100)),
        ]
        if not enabled:
            labels.append(('HifChoiceUnselected', 'スキルカードが選択されていません', (120, 200, 480, 90)))
        for node, text, roi in labels:
            rows = _inventory_ocr(context, frame, node, roi=roi)
            row = _unique_inventory_text(rows, text, roi, threshold=.95)
            anchors.append(dict(text=row.text, box=tuple(row.box), score=float(row.score)))
    title_rows = ()
    if enabled and selected_slot is not None:
        rows = _choice_selected_title_rows(context, frame)
        if (len(rows) != 1 or type(rows[0].score) not in (float, int)
                or not math.isfinite(rows[0].score) or not (0 if upstream else .95) <= rows[0].score <= 1
                or not rows[0].text.strip()):
            raise ValueError('HIF choice selected detail title is unresolved')
        title_rows = tuple(dict(text=row.text, score=float(row.score), box=tuple(row.box)) for row in rows)
    return dict(title='移動', selected_count=int(enabled), selected_slot=selected_slot,
                selected_count_source=('enabled_move_button' if enabled and selected_slot is None
                                       else 'four_corner_brackets' if enabled else 'unselected_prompt'),
                unselected=not enabled, detail_title_rows=title_rows,
                confirm_enabled=enabled, confirm_target=confirm_target,
                boxes=boxes, anchors=anchors, scroll_offset=scroll_offset)


def bind_hif_choice_candidates(projected, observed_card_ids, boxes, *, badge_presence):
    """Bind the complete observed multiset; equivalent copies stay in one zone.

The UI does not label origins. Same-id copies from different zones, permanent
variants, or known draw positions therefore require more evidence and stop.
"""
    state = projected.pending_state
    pending = state.pending_choice
    if (pending is None or pending.action_kind is not ActionKind.MOVE_CARD
            or pending.source_id != 'drink:22' or pending.selection_optional
            or pending.selections_remaining != 1):
        raise ValueError('HIF choice is not the supported one-card black-vinegar retrieval')
    ids, boxes = tuple(observed_card_ids), tuple(tuple(box) for box in boxes)
    if (not ids or any(not isinstance(card_id, str) or not card_id for card_id in ids)
            or len(boxes) != len(ids) or len(set(boxes)) != len(boxes)
            or any(len(box) != 4 or any(type(value) is not int for value in box)
                   or min(box[:2]) < 0 or min(box[2:]) <= 0 for box in boxes)):
        raise ValueError('HIF choice physical candidate identities are incomplete')
    cards = {card.instance_id: card for card in state.deck.all_cards}
    expected_ids = tuple(pending.candidate_instance_ids)
    if (len(set(expected_ids)) != len(expected_ids)
            or any(instance not in cards for instance in expected_ids)):
        raise ValueError('HIF choice model candidates are incomplete or duplicated')
    bundle = projected.source_bundle
    permanent = {instance: (canonical, customizations)
                 for instance, canonical, customizations in bundle.inventory_keys}
    if (len(permanent) != len(bundle.inventory_keys) or set(permanent) != set(cards)
            or Counter(ids) != Counter(permanent[instance][0] for instance in expected_ids)):
        raise ValueError('HIF choice observed candidates differ from the complete pending set')
    retained = retained_permanent_card_rows(bundle, state, {'choice': ids})
    presence = tuple(badge_presence)
    if (len(presence) != len(ids)
            or any(value != ('DETAIL_CANDIDATE' if row['customizations'] else 'CONFIDENT_ZERO')
                   for value, row in zip(presence, retained))):
        raise ValueError('HIF choice customization badge differs from verified permanent identity')
    zones = {}
    for zone in ('draw_pile', 'discard'):
        for card in getattr(state.deck, zone):
            if card.instance_id in zones:
                raise ValueError('HIF choice candidate occupies multiple source zones')
            zones[card.instance_id] = zone
    if any(instance not in zones for instance in expected_ids):
        raise ValueError('HIF choice candidate source is not draw or discard')
    by_id = defaultdict(list)
    for instance in expected_ids:
        by_id[permanent[instance][0]].append(instance)
    for instances in by_id.values():
        keys = {(cards[instance].semantic_key, zones[instance]) for instance in instances}
        if len(keys) != 1:
            raise ValueError('HIF choice same-id variants or source zones are visually ambiguous')
        if len(instances) > 1 and zones[instances[0]] == 'draw_pile':
            known = state.deck.draw_pile[:state.deck.known_top_count]
            if (state.deck.knowledge is DeckKnowledge.KNOWN_ORDER
                    or any(card.instance_id in instances for card in known)):
                raise ValueError('HIF choice ordered duplicate requires positional evidence')
        instances.sort()
    used = Counter()
    candidates = []
    for slot, (card_id, box) in enumerate(zip(ids, boxes)):
        equivalent = by_id[card_id]
        instance = equivalent[used[card_id]]
        used[card_id] += 1
        candidates.append(dict(slot=slot, box=box, card_id=card_id, instance_id=instance,
                               model_card_id=cards[instance].card_id,
                               customizations=retained[slot]['customizations'],
                               equivalent_instance_ids=tuple(equivalent),
                               source_zone=zones[instance]))
    return dict(candidates=candidates,
                card_slots=tuple((row['instance_id'], row['slot']) for row in candidates))


def confirm_hif_choice_title(catalog, text, *, candidate_card_ids):
    """Keep the observed HIF glyph variant within one complete family title."""
    candidate_card_ids = tuple(candidate_card_ids)
    if '顏' in text:
        # 1763 reads the complete title with U+984F; the fixed catalog uses
        # U+9854. Do not extend the shared arena aliases or accept a fragment.
        normalized = text.replace('顏', '顔')
        matches = [card_id for card_id in candidate_card_ids
                   if normalized == catalog.skill_card_title(card_id)]
        if len(matches) != 1:
            raise ValueError('HIF choice title glyph variant has no unique complete family title')
        text = normalized
    return catalog.confirm_clicked_skill_card_candidates(text, candidate_card_ids=candidate_card_ids)


def resolve_hif_choice_detail_evidence(catalog, recognizer, projected, *, slot, box,
                                      visual_group, frame_ids, title_rows,
                                      observation_epoch, capture_monotonic, upstream=False):
    """Resolve a caller-proven selected slot using the observed detail title.

    The caller must independently prove which thumbnail is selected in those
    frames. This function neither selects a card nor infers selection from text.
    The evidence remains tied to this pending run and immutable grid position.
    """
    frame_ids, title_rows = tuple(frame_ids), tuple(tuple(rows) for rows in title_rows)
    capture_monotonic = tuple(capture_monotonic)
    count = len(frame_ids)
    if (type(slot) is not int or not 0 <= slot < len(CHOICE_BOXES) or tuple(box) != CHOICE_BOXES[slot]
            or count not in (1, 2) or len(set(frame_ids)) != count
            or any(not isinstance(value, str) or not value for value in frame_ids)
            or len(title_rows) != count or not isinstance(observation_epoch, str)
            or not observation_epoch or len(capture_monotonic) != count
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in capture_monotonic)
            or any(left >= right for left, right in zip(capture_monotonic, capture_monotonic[1:]))):
        raise ValueError('HIF choice detail is not bound to its selected-slot frames')
    gallery = recognizer.recognizer.gallery
    candidate_ids = tuple(sorted({int(gallery.card_ids[index])
        for index in recognizer.groups.get(visual_group, ())}))
    ids = []
    serialized = []
    for rows in title_rows:
        values = tuple(dict(row) if isinstance(row, dict) else dict(
            text=row.text, score=row.score, box=tuple(row.box)) for row in rows)
        if (not values or any(type(row.get('score')) not in (int, float)
                or not math.isfinite(row['score']) or not (0 if upstream else .95) <= row['score'] <= 1
                or not isinstance(row.get('text'), str) or not row['text'] for row in values)):
            raise ValueError('HIF choice detail title is unresolved')
        text = '\n'.join(row['text'] for row in values)
        ids.append(str(confirm_hif_choice_title(catalog, text, candidate_card_ids=candidate_ids)))
        serialized.append(values)
    if len(set(ids)) != 1:
        raise ValueError('HIF choice detail title changes across frames')
    return dict(slot=slot, box=tuple(box), visual_group=visual_group, card_id=ids[0],
                run_id=projected.source_bundle.run_id, revision=projected.source_bundle.revision,
                observation_epoch=observation_epoch, capture_monotonic=capture_monotonic,
                frame_ids=frame_ids, title_rows=tuple(serialized),
                source='miyu_move_title')


def observe_hif_choice_panel(context, frames, recognizer, projected, *, frame_ids,
                             detail_evidence=(), detail_catalog=None, detail_epoch=None):
    """Bind the cards visible on the current retrieval page."""
    frames, frame_ids = tuple(frames), tuple(frame_ids)
    if len(frames) != 1 or len(frame_ids) != 1 or not isinstance(frame_ids[0], str) or not frame_ids[0]:
        raise ValueError('HIF choice requires one current page and its frame identity')
    frame, = frames
    pending = getattr(getattr(projected, 'pending_state', None), 'pending_choice', None)
    expected_count = len(pending.candidate_instance_ids) if pending is not None else None
    page = read_hif_choice_panel(context, frame, expected_count=expected_count)
    boxes, selected_slot = page['boxes'], page.get('selected_slot')
    pending = projected.pending_state.pending_choice
    if (pending is None or pending.action_kind is not ActionKind.MOVE_CARD
            or pending.source_id != 'drink:22' or pending.selection_optional
            or pending.selections_remaining != 1
            or len(pending.candidate_instance_ids) != len(boxes)
            or len(set(pending.candidate_instance_ids)) != len(boxes)):
        raise ValueError('HIF choice visible grid differs from the pending retrieval boundary')
    identities = []
    for box in boxes:
        identity = recognizer.identify(frame, box, plan='sense')
        identities.append(dict(box=box, observations=(identity,), card_id=identity.card_id))
    details = tuple(detail_evidence)
    if len({row.get('slot') for row in details}) != len(details):
        raise ValueError('HIF choice detail slot evidence is duplicated')
    for detail in details:
        slot = detail.get('slot')
        if (detail_catalog is None or type(slot) is not int or not 0 <= slot < len(boxes)
                or detail.get('run_id') != projected.source_bundle.run_id
                or detail.get('revision') != projected.source_bundle.revision
                or not detail_epoch or detail.get('observation_epoch') != detail_epoch
                or detail.get('source') != 'miyu_move_title'
                or tuple(detail.get('box', ())) != boxes[slot]):
            raise ValueError('HIF choice detail evidence belongs to another pending grid')
        groups = {row.visual_group for row in identities[slot]['observations']}
        if groups != {detail.get('visual_group')} or None in groups:
            raise ValueError('HIF choice detail no longer matches the visible family')
        checked = resolve_hif_choice_detail_evidence(detail_catalog, recognizer, projected,
            slot=slot, box=boxes[slot], visual_group=detail['visual_group'],
            frame_ids=detail.get('frame_ids', ()), title_rows=detail.get('title_rows', ()),
            observation_epoch=detail_epoch, capture_monotonic=detail.get('capture_monotonic', ()),
            upstream=True)
        if (detail.get('source') != checked['source'] or detail.get('card_id') != checked['card_id']
                or identities[slot]['card_id'] not in (None, checked['card_id'])):
            raise ValueError('HIF choice detail disagrees with visible card identity')
        identities[slot]['card_id'] = checked['card_id']
        identities[slot]['detail_evidence'] = checked
    ids = tuple(row['card_id'] for row in identities)
    unresolved = tuple(index for index, card_id in enumerate(ids) if card_id is None)
    badge_presence = tuple(stable_customization_badge_presence(
        (measure_customization_badge(frame, box),), required_frames=1).state.value for box in boxes)
    result = dict(accepted=False, frame_ids=frame_ids, single_frame=True,
                title='移動', unselected=selected_slot is None, selected_count=page['selected_count'],
                selected_slot=selected_slot, selection_limit=1,
                selected_count_source='four_corner_brackets' if selected_slot is not None else 'unselected_prompt',
                confirm_enabled=selected_slot is not None, confirm_target=page['confirm_target'],
                unresolved_slots=unresolved, badge_presence=badge_presence,
                pages=(page,), identities=tuple(dict(box=row['box'], card_id=row['card_id'],
                    detail_evidence=row.get('detail_evidence'),
                    observations=tuple(asdict(value) for value in row['observations'])) for row in identities))
    if unresolved:
        result['preview_targets'] = tuple(dict(slot=slot, box=boxes[slot]) for slot in unresolved)
        return result
    binding = bind_hif_choice_candidates(projected, ids, boxes, badge_presence=badge_presence)
    if selected_slot is not None:
        if detail_catalog is None:
            raise ValueError('HIF choice selected detail has no fixed identity catalog')
        candidate = binding['candidates'][selected_slot]
        confirm_hif_choice_title(detail_catalog,
            '\n'.join(row['text'] for row in page['detail_title_rows']),
            candidate_card_ids=(int(candidate['card_id']),))
        result['selected_instance_id'] = candidate['instance_id']
    result.update(accepted=True, **binding)
    return result
