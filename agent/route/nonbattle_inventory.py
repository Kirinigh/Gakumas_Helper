"""Observe the flat nonbattle card list with the existing identity readers."""
from __future__ import annotations

import re
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
from dataclasses import asdict

import numpy as np
from utils.runtime_paths import resolve_runtime_path


def bind_inventory_census(producer, parent_scope, retained=None):
    """Keep business observations within one parent and actual inventory opening.

    Registration changes while scrolling; its first opening stays fixed. A cold
    continuation restores that same opening rather than starting navigation again.
    """
    parent = getattr(producer, 'parent', producer)
    host = parent.host
    scope = dict(run_id=host.run_id, game_pid=host.game_pid,
        ownership_revision=host.ownership['revision'], parent=parent_scope)
    previous = getattr(producer, 'inventory_scope', None)
    saved = (retained or {}).get('census_session')
    gate = host.hooks.transition_gate
    opening = gate.pending is not None and gate.pending.target.kind == 'recovery_inventory'
    if opening:
        epoch = [host.clock.epoch, gate.issued_at_ms, gate.pending.source.revision]
    elif saved is not None and saved['scope'] == scope:
        epoch = saved['opening']
    elif previous is not None and previous['scope'] == scope:
        epoch = previous['opening']
    elif retained and retained.get('frame_ids'):
        epoch = ['registered', retained['frame_ids'][0]]
    else:
        epoch = ['observed', host.clock.now()]
    session = dict(scope=scope, opening=epoch)
    if previous != session:
        producer.inventory = None
        producer.inventory_scope = session
        if hasattr(producer, 'complete_ledger'):
            producer.inventory_navigation = producer.complete_ledger = None
    if saved != session and (saved is not None or opening):
        return None
    return retained


def retain_inventory_registration(host, page, inventory):
    """Persist this accepted census view as a future read-only registration."""
    result = deepcopy(inventory)
    if host._input_source is not None:
        rows = host._input_source(page)
        if [row.get('frame_id') for row in rows] != list(page.frame_ids):
            raise ValueError('inventory registration does not contain the current source frames')
        result['durable_frames'] = deepcopy(rows)
        result['durable_frame_ids'] = list(page.frame_ids)
    return result


def load_flat_inventory_readers():
    from card_selection import EmbeddingCardRecognizer
    from live_decision.inventory import InventoryThumbnailRecognizer
    from arena_winrate.badge_reference import BadgeReferenceGallery
    from live_decision.inventory_customization import load_customization_catalog
    reader = InventoryThumbnailRecognizer(
        BadgeReferenceGallery.load(resolve_runtime_path(Path('assets/resource/base/model/embedding/arena_badge_reference'),
            Path('resource/base/model/embedding/arena_badge_reference'))),
        EmbeddingCardRecognizer.load(resolve_runtime_path(Path('assets/resource/base/model/classify/card_embedding'),
            Path('resource/base/model/classify/card_embedding'))))
    catalog = load_customization_catalog(resolve_runtime_path(Path('assets/data/skill_cards.csv'), Path('data/skill_cards.csv')),
        resolve_runtime_path(Path('assets/data/customizations.json'), Path('data/customizations.json')))
    return reader, catalog


def flat_inventory_count(rows):
    counts = [int(match[1]) for row in rows
        if row['score'] >= .95 and 65 <= row['box'][0] and 1010 <= row['box'][1]
        and row['box'][0] + row['box'][2] <= 340 and row['box'][1] + row['box'][3] <= 1085
        and (match := re.fullmatch(r'スキルカード\((\d{1,2})\)', row['text']))]
    if len(counts) != 1 or not 1 <= counts[0] <= 99:
        raise ValueError('flat inventory total is not uniquely observed')
    return counts[0]


def flat_inventory_is_at_top(image):
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        return False
    # Current 480 profile: the three opaque inner columns, excluding the
    # anti-aliased outer column and the transparent moving stage behind it.
    strip = image[395:1010, 702:705].astype('int16')
    handle = (np.abs(strip - np.array([146, 143, 134])).max(2) <= 4).mean(1) >= .75
    positions = np.flatnonzero(handle)
    runs = [run for run in np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1) if len(run)]
    complete = [run for run in runs if len(run) >= 60]
    # 152's real handle is continuous from 404 to 932. Moving stage pixels
    # below it independently match the colour for one row (989). They cannot
    # split, extend or stand in for the unique complete opaque handle.
    return bool(len(complete) == 1 and 402 <= complete[0][0] + 395 <= 406
        and all(len(run) <= 4 or run is complete[0] for run in runs))


def flat_inventory_boxes(total, displacement):
    # On a count-less ordinary review these are just thumbnail candidates in
    # the viewport. Only independently identified artwork becomes a known slot.
    if total is None:
        total = 4 * (1 + (1010 - 142 - 406 + displacement) // 154)
    return {index: ((48, 208, 368, 529)[index % 4], 406 + index // 4 * 154 - displacement, 142, 142)
        for index in range(total)
        if 406 <= 406 + index // 4 * 154 - displacement <= 1010 - 142
        and not (index % 4 == 3 and 406 + index // 4 * 154 - displacement + 142 > 947)}


def flat_scroll_measurement_matches(source, receipt):
    """Consume the existing unique, opaque-art registration; not swipe distance.

    inventory_scroll_displacement already searches through 450px and requires
    .97 on every surviving patch across independent artwork columns. An 80px
    input drag is not a 120px maximum on the content's measured movement.
    """
    registration = receipt.get('registration') or {}
    shifts = registration.get('shifts', ())
    if (len(shifts) != 1
            or any(type(shift) is not int or not 0 < shift <= 450 for shift in shifts)
            or registration.get('source_displacement') != source.get('displacement')
            or receipt.get('displacement') != source.get('displacement', -1000) + shifts[0]
            or receipt.get('total') != source.get('total')):
        return False
    old = {row['index']: row for row in source.get('slots', ())}
    anchors = registration.get('anchor_indices', ())
    if (len(anchors) < 2 or len(set(anchors)) != len(anchors) or any(index not in old for index in anchors)
            or any(not isinstance(old[index].get('box'), (list, tuple)) or len(old[index]['box']) != 4 for index in anchors)):
        return False
    for shift in shifts:
        columns, visible = set(), 0
        for index in anchors:
            x, y, width, height = old[index]['box']
            target = y - shift
            top = max(414, target + 8)
            bottom = min(999, target + height - 8, 939 if x + width > 606 else 999)
            if bottom - top >= 48:
                columns.add(x)
                visible += 1
        if visible < 2 or len(columns) < 2:
            return False
    return True


def align_flat_inventory_rows(frames, boxes):
    """Use the same two-tile border measurement on a single inventory image."""
    if len(frames) != 1:
        raise ValueError("flat inventory geometry requires one current image")
    import cv2
    if frames[0].shape != (1280, 720, 3):
        return boxes
    edges = cv2.Canny(cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY), 80, 180) > 0
    aligned = dict(boxes)
    for y in {box[1] for box in boxes.values()}:
        tiles = [(index, x) for index, (x, top, w, h) in boxes.items() if top == y and (w, h) == (142, 142)]
        offsets = {}
        for _, x in tiles:
            for top in range(max(414, y - 2), min(999 - 138, y + 8) + 1):
                bottom = top + 138
                if (edges[top, x+12:x+130].mean() >= .95 and edges[bottom, x+12:x+130].mean() >= .95
                        and edges[top+8:bottom-8, x-3:x+3].any(axis=1).mean() >= .95
                        and edges[top+8:bottom-8, x+137:x+144].any(axis=1).mean() >= .95):
                    offsets[top-2] = offsets.get(top-2, 0) + 1
        if len(offsets) != 1:
            continue
        corrected, matches = next(iter(offsets.items()))
        if (matches < 2 or abs(corrected-y) > 4 or corrected < 406 or corrected+142 > 1007
                or any(x+142 > 606 and corrected+142 > 947 for _, x in tiles)):
            continue
        for index, x in tiles:
            aligned[index] = (x, corrected, 142, 142)
    return aligned


def read_course_thumbnail(reader, image, box, *, plan='sense'):
    card = reader.identify(image, box, plan=plan)
    return dict(card_id=card.card_id, observations=(card,))


def read_course_badge(image, slot):
    from arena_winrate.customization_badge import measure_customization_badge, stable_customization_badge_presence
    zone, index, card_id, box = slot
    measurements = (measure_customization_badge(image, box),)
    decision = stable_customization_badge_presence(measurements, required_frames=1)
    if decision.state.value == 'AMBIGUOUS':
        raise ValueError('inventory customization badge is ambiguous')
    return dict(zone=zone, index=index, card_id=card_id, box=box, presence=decision.state.value,
        reason=decision.reason, observations=measurements)


def _course_detail_title(catalog, card_id, rows):
    from live_decision.upgrade import normalize_upgrade_title
    from live_decision.inventory import INVENTORY_DETAIL_TITLE_ROI

    from .nonbattle_observation import _within
    titles = [row for row in rows if _within(row.box, INVENTORY_DETAIL_TITLE_ROI)]
    if len(titles) != 1 or not .95 <= titles[0].score <= 1:
        raise ValueError('inventory detail title is unresolved')
    title = titles[0].text
    if normalize_upgrade_title(title) != catalog.skill_card_title(int(card_id)):
        raise ValueError('inventory detail title differs from its verified card')
    catalog.confirm_clicked_skill_card(title, predicted_card_id=int(card_id))
    return title


def _course_detail_badge(context, image, frame_id):
    from live_decision.inventory import _inventory_ocr
    from arena_winrate.customization_badge import measure_customization_badge
    from live_decision.inventory_detail_badge import DETAIL_CARD_BOX, _plate, _frame_digit, _profile_canvases
    measured = measure_customization_badge(image, DETAIL_CARD_BOX)
    _, canvases = _plate(measured)
    measured.update(frame_id=frame_id, image_shape=list(image.shape), render_profile='nonbattle_white', ocr_variants=[])
    for index, canvas in _profile_canvases(canvases, 'nonbattle_white'):
        rows = _inventory_ocr(context, canvas, 'LiveInventoryDetailBadgeGlyph', expected=r'^[1-3]$')
        measured['ocr_variants'].append(dict(index=index, shape=list(canvas.shape),
            rows=[dict(text=row.text.strip(), score=row.score, box=list(row.box)) for row in rows]))
    measured['count'], _ = _frame_digit(measured)
    return dict(frame_ids=[frame_id], card_box=list(DETAIL_CARD_BOX), observations=[measured], count=measured['count'])


def resolve_nonbattle_detail(context, catalog, card_id, observations, frames, frame_ids):
    import unicodedata
    if len(frames) != 1 or len(frame_ids) != 1 or len(observations) != 1:
        raise ValueError('nonbattle inventory detail requires one current image and receipt')
    title = _course_detail_title(catalog, card_id, observations[0])
    if '++' in unicodedata.normalize('NFKC', title):
        raise ValueError('temporary support tier does not identify permanent customizations')
    ordered = sorted(observations[0], key=lambda value: (value.box[1], value.box[0]))
    text = ordered[0].text + ''.join(value.text for value in ordered[1:])
    badge = _course_detail_badge(context, frames[0], frame_ids[0])
    catalog.confirm_clicked_skill_card(text, predicted_card_id=int(card_id))
    groups = catalog.resolve_effective_customizations(int(card_id), text, expected_count=badge['count'],
        allow_positive_completion=False)
    return groups, badge


class FlatInventoryCollector:
    """Keep physical positions distinct, including duplicate business IDs.

    This collects evidence only. The route host owns every detail click and
    measured scroll; neither the collector nor its readers can issue input.
    """
    def __init__(self, reader, catalog):
        self.reader, self.catalog = reader, catalog
        self.total = None
        self.displacement = 0
        self.previous = None
        self.previous_boxes = {}
        self.boxes = {}
        self.slots = {}
        self.frame_ids = []
        self.detail_rejection = None
        self.detail_wait_reason = None
        self.selected_index = None
        self.previous_frame_ids = ()
        self.registration = None

    @staticmethod
    def _supported_geometry_anchors(previous, current, anchors):
        """Select readable geometry with the existing opaque-art registration.

        Business IDs and badges are deliberately absent here. The unchanged
        registration primitive must still reconcile every supported anchor.
        """
        import cv2
        previous = cv2.GaussianBlur(previous,(1,5),1.)
        current = cv2.GaussianBlur(current,(1,5),1.)
        proposals, readable = set(), {}
        for index, (x,y,w,h) in anchors.items():
            patch = previous[y+h-56:y+h-8,x+8:x+w-8]
            if float(patch.std()) < 10:
                continue
            scores = cv2.matchTemplate(current[414:999,x+8:x+w-8],patch,cv2.TM_CCOEFF_NORMED)[:,0]
            for position in np.flatnonzero(scores >= .97):
                delta = y+h-56-(414+int(position))
                if -2 <= delta <= 450:
                    proposals.add(delta)
            readable[index] = (x,y,w,h)
        supported = {}
        for index, (x,y,w,h) in readable.items():
            for delta in sorted(proposals):
                target = y-delta
                top, bottom = max(414,target+8), min(999,target+h-8,939 if x+w>606 else 999)
                if bottom-top < 48:
                    continue
                reference = previous[top+delta:bottom+delta,x+8:x+w-8]
                observed = current[top:bottom,x+8:x+w-8]
                if float(cv2.matchTemplate(observed,reference,cv2.TM_CCOEFF_NORMED)[0,0]) >= .97:
                    supported[index] = (x,y,w,h)
                    break
        return supported

    def observe(self, context, frames, rows_by_frame, frame_ids, *, inspect_index=None,
            allow_unknown_badges=False, allow_unknown_total=False, reread_top=False):
        fields = ('total', 'displacement', 'previous', 'previous_boxes', 'boxes', 'slots', 'frame_ids',
            'detail_rejection', 'detail_wait_reason', 'selected_index', 'previous_frame_ids', 'registration')
        before = {name: getattr(self, name) for name in fields}
        self.slots, self.frame_ids = deepcopy(self.slots), list(self.frame_ids)
        try:
            options = dict(inspect_index=inspect_index, allow_unknown_badges=allow_unknown_badges,
                allow_unknown_total=allow_unknown_total)
            try:
                return self._observe(context, frames, rows_by_frame, frame_ids, **options)
            except ValueError as error:
                if (str(error) != 'inventory scroll has no unique overlapping translation' or not reread_top
                        or inspect_index is not None or self.selected_index is not None
                        or self.displacement != 0 or self.registration is not None or len(self.frame_ids) != 1
                        or not flat_inventory_is_at_top(self.previous) or not flat_inventory_is_at_top(frames[0])):
                    raise
                # The opening fade can distort the first artwork read. No
                # navigation has occurred between these observations of the
                # same top. Keep independent detail proofs and reread the rest.
                self.previous = None
                self.slots = {index: row for index, row in self.slots.items()
                    if row.get('detail') or row.get('identity_retained')}
                return self._observe(context, frames, rows_by_frame, frame_ids, **options)
        except BaseException:
            # A failed new badge/identity batch must not keep a measured shift
            # and then add it again on the next independent read (156).
            for name, value in before.items():
                setattr(self, name, value)
            raise

    def _observe(self, context, frames, rows_by_frame, frame_ids, *, inspect_index=None,
            allow_unknown_badges=False, allow_unknown_total=False):
        from live_decision.inventory import inventory_scroll_displacement

        from .engine import has_single_frame_id
        if len(frames) != 1 or len(rows_by_frame) != 1 or not has_single_frame_id(frame_ids):
            raise ValueError('flat inventory requires one current frame')
        image = frames[0]
        try:
            total = flat_inventory_count(rows_by_frame[0])
        except ValueError:
            if not allow_unknown_total:
                raise
        else:
            if self.total is not None and self.total != total:
                raise ValueError('flat inventory total changed')
            self.total = total
        # Opening a detail enlarges that thumbnail. It cannot serve as a
        # fixed-scale scroll anchor or replace its earlier plain-tile proof.
        excluded = {index for index in (inspect_index, self.selected_index) if index is not None}
        weak_geometry = set()
        if self.previous is None:
            if not flat_inventory_is_at_top(image):
                raise ValueError('flat inventory collection must begin at the observed top')
        else:
            anchors = {index: box for index, box in self.previous_boxes.items() if index not in excluded}
            supported = anchors
            try:
                shift = inventory_scroll_displacement(self.previous, image, anchors)
            except ValueError as error:
                if str(error) not in ('inventory scroll has no unique overlapping translation',
                        'inventory scroll anchor lacks visible artwork'):
                    raise
                supported = self._supported_geometry_anchors(self.previous, image, anchors)
                shift = inventory_scroll_displacement(self.previous, image, supported)
                weak_geometry = set(anchors) - set(supported)
            self.registration = dict(source_frame_ids=list(self.previous_frame_ids),
                receipt_frame_ids=list(frame_ids), shifts=[shift], source_displacement=self.displacement,
                excluded_indices=sorted(excluded), anchor_indices=sorted(supported))
            if supported is not anchors:
                self.registration['geometry_anchor_indices_by_frame'] = [sorted(supported)]
                self.registration['weak_geometry_indices_by_frame'] = [sorted(weak_geometry)]
            self.displacement += shift
        boxes = align_flat_inventory_rows(frames, flat_inventory_boxes(self.total, self.displacement))
        if len(boxes) < 2 or len({box[0] for box in boxes.values()}) < 2:
            raise ValueError('flat inventory lacks two independent artwork columns')
        readable = {index: box for index, box in boxes.items() if index not in excluded}
        # Registration has matched every remaining opaque card patch in the
        # current observation at .97. A zero shift with the
        # exact same physical box preserves an already-observed identity;
        # badge state and opened detail are still read from the new frames.
        retained = {index for index, box in readable.items()
            if self.registration is not None and self.registration['shifts'] == [0]
            and index in self.previous_boxes and tuple(box) == tuple(self.previous_boxes[index])
            and index not in weak_geometry
            and self.slots.get(index, {}).get('card_id') is not None
            and self.slots[index].get('identities')}
        new = {index: box for index, box in readable.items() if index not in retained}
        batch = [read_course_thumbnail(self.reader, image, box) for box in new.values()]
        observed = dict(zip(new, batch, strict=True))
        slots = dict(self.slots)
        for index, box in readable.items():
            old = slots.get(index)
            identity = observed.get(index)
            card_id = old['card_id'] if index in retained else identity['card_id']
            if self.total is None and old is None and card_id is None:
                continue
            if old is not None and old['card_id'] is not None and card_id is not None and old['card_id'] != card_id:
                raise ValueError('flat inventory physical card changed identity')
            try:
                slot = ('flat', index, card_id, box)
                presence = read_course_badge(image, slot)
            except ValueError as error:
                if not allow_unknown_badges or str(error) != 'inventory customization badge is ambiguous':
                    raise
                presence = dict(presence='UNKNOWN', observations=[])
            if old is not None and old['presence'] != presence['presence']:
                raise ValueError('flat inventory permanent badge changed')
            row = dict(old or {}, index=index, card_id=card_id or (old or {}).get('card_id'),
                current_card_id=card_id,
                presence=presence['presence'], box=list(box), frame_ids=list(frame_ids),
                identities=old['identities'] if index in retained else [asdict(value) for value in identity['observations']],
                badge_observations=presence['observations'])
            if index in weak_geometry:
                row['identity_needs_detail'] = card_id is None
            elif index not in retained and card_id is not None:
                row.pop('identity_needs_detail',None)
            if index in retained:
                row['identity_retained'] = dict(
                    source_frame_ids=(old.get('identity_retained') or {}).get('source_frame_ids', old['frame_ids']),
                    receipt_frame_ids=list(frame_ids), registration=self.registration)
            if presence['presence'] == 'CONFIDENT_ZERO':
                row['customizations'] = {}
            slots[index] = row
        if self.total is None:
            boxes = {index: box for index, box in boxes.items() if index in slots}
        self.slots, self.boxes = slots, boxes
        self.previous, self.previous_boxes = image.copy(), boxes
        self.previous_frame_ids = tuple(frame_ids)
        self.frame_ids.extend(frame_ids)
        if inspect_index is not None:
            self._inspect_detail(context, frames, rows_by_frame, frame_ids, inspect_index)
        return self.record()

    def revalidate_retained_details(self, context, load_frame):
        """Reinterpret retained OCR against its original independent images."""
        for row in self.slots.values():
            detail = row.get('detail')
            if row['presence'] == 'CONFIDENT_ZERO' or not detail or detail.get('nonbattle_schema') == 1:
                continue
            keys = detail['frame_ids']
            frames = [load_frame(key) for key in keys]
            observations = [tuple(SimpleNamespace(**value) for value in group) for group in detail['rows']]
            groups, badge = resolve_nonbattle_detail(context, self.catalog, row['card_id'], observations, frames, keys)
            old = row['customizations']
            row['customizations'] = {str(key): count for key, count in groups.items()}
            row['detail'] = dict(detail, nonbattle_schema=1, badge_count_evidence=badge,
                original_partial_customizations=old)

    def _inspect_detail(self, context, frames, rows_by_frame, frame_ids, index):
        from live_decision.inventory_customization import read_inventory_detail_text_rows
        if len(frames) != 1 or len(rows_by_frame) != 1 or len(frame_ids) != 1:
            raise ValueError('inventory detail requires one current frame')
        if index not in self.boxes or index not in self.slots:
            raise ValueError('flat inventory detail physical target disappeared')
        row = self.slots[index]
        self.detail_wait_reason = None
        # The exact same images already have full Maa OCR. Reuse those rows
        # with their original boxes/confidence; crop rereads still use Maa.
        cached = {id(frame): tuple(SimpleNamespace(**value) for value in rows)
            for frame, rows in zip(frames, rows_by_frame, strict=True)}
        def recognize(name, image, **kwargs):
            if name == 'LiveInventoryCustomizationDetail' and id(image) in cached:
                return SimpleNamespace(hit=True, filtered_results=cached[id(image)])
            return context.run_recognition(name, image, **kwargs)
        cached_context = SimpleNamespace(run_recognition=recognize)
        observations = [read_inventory_detail_text_rows(cached_context, frame) for frame in frames]
        def read_title(card_id):
            return _course_detail_title(self.catalog, card_id, observations[0])
        expected = row['card_id']
        if expected is None:
            possible = set()
            for card_id in self.catalog._cards_by_id:
                try:
                    read_title(str(card_id))
                    possible.add(str(card_id))
                except ValueError:
                    continue
            if len(possible) != 1:
                # The old detail/unselected panel can remain during the
                # original click's transition. It cannot settle that click.
                self.detail_wait_reason = 'flat detail title has no unique current card identity'
                return
            expected = possible.pop()
        try:
            title = read_title(expected)
        except ValueError as error:
            self.detail_wait_reason = str(error)
            return
        if any(len(group) < 2 for group in observations):
            # 154's first post-click frame has the correct title while the
            # effect body is still empty. Keep the original pending/current
            # detail target; another complete batch must resolve every effect.
            self.detail_wait_reason = 'flat detail title is visible but its complete effect body is not'
            return
        try:
            # The current body is authoritative for permanent options. Zero
            # badge proof already suffices for plain cards without options.
            customizations, badge = (resolve_nonbattle_detail(context, self.catalog, expected,
                observations, frames, frame_ids) if row['presence'] != 'CONFIDENT_ZERO' else ({}, None))
        except ValueError as error:
            self.detail_rejection = str(error)
            return
        row.update(card_id=expected, current_card_id=expected, identity_needs_detail=False,
            customizations={str(k): v for k, v in customizations.items()},
            detail=dict(title=title, frame_ids=list(frame_ids), nonbattle_schema=1, badge_count_evidence=badge,
                rows=[[dict(text=value.text, box=list(value.box), score=value.score)
                    for value in group] for group in observations]))
        self.detail_rejection = None
        self.selected_index = index

    @property
    def complete(self):
        return (self.total is not None and set(self.slots) == set(range(self.total))
            and all(row['card_id'] is not None and not row.get('identity_needs_detail',False) and 'customizations' in row
                and (row['presence'] == 'CONFIDENT_ZERO' or row.get('detail', {}).get('nonbattle_schema') == 1)
                for row in self.slots.values()))

    def next_navigation(self):
        if self.detail_rejection is not None:
            raise ValueError('flat inventory opened detail could not be parsed: ' + self.detail_rejection)
        for index, box in self.boxes.items():
            row = self.slots[index]
            if (row['card_id'] is None or row.get('identity_needs_detail',False) or 'customizations' not in row
                    or row['presence'] != 'CONFIDENT_ZERO' and row.get('detail', {}).get('nonbattle_schema') != 1):
                return dict(kind='recovery_inventory_detail', key=f'inventory-detail:{index}',
                    target=box, inspect_index=index)
        if self.complete:
            return None
        if all(index in self.slots for index in range(self.total)):
            raise ValueError('flat inventory unresolved cards are outside the current viewport')
        return dict(kind='recovery_inventory_scroll', key=f'inventory-scroll:{self.displacement}',
            target=(350, 770, 1, 80))

    def record(self):
        return dict(total=self.total, displacement=self.displacement,
            visible_boxes={str(index): list(box) for index, box in self.boxes.items()},
            selected_index=self.selected_index,
            registration=self.registration,
            slots=[self.slots[index] for index in sorted(self.slots)], complete=self.complete,
            frame_ids=list(dict.fromkeys(self.frame_ids)), detail_rejection=self.detail_rejection,
            **({'census_session': deepcopy(self.census_session)} if hasattr(self, 'census_session') else {}),
            **({'detail_wait_reason': self.detail_wait_reason} if self.detail_wait_reason is not None else {}))
