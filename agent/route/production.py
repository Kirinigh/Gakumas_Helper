"""Bind authorised Maa capture/normal UI to the persistent HIF route host.

The caller supplies real permission callbacks, never permission from task JSON.
Recovery navigation is nonconsuming and has its own explicit scene whitelist.
"""
from __future__ import annotations

import re
import json
import time
import unicodedata
from copy import deepcopy
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
from utils import logger
from utils.local_data import LocalDataError, get_runtime_store
from utils.runtime_paths import resolve_runtime_path

from .host import HifRouteHost, HifRoutePage, windows_route_clock
from .engine import Target, Outcome, Decision, Observation, propose_unique, source_frames_match, validate_observation
from .footer import cultivation_menu_target as interval_menu_target

_ROUTE_SCREENSHOT_ROLE = 'hif_route_input_source'


def _row(rows, pattern, roi, *, minimum=.95):
    x, y, w, h = roi
    matches = [row for row in rows if row['score'] >= minimum
        and re.fullmatch(pattern, row['text']) and x <= row['box'][0] and y <= row['box'][1]
        and row['box'][0] + row['box'][2] <= x + w and row['box'][1] + row['box'][3] <= y + h]
    if len(matches) != 1:
        raise ValueError('HIF production field is not uniquely observed: ' + pattern)
    return matches[0]


def _text(rows, pattern, roi):
    return _row(rows, pattern, roi)['text']


def recognize_route_inventory_button(context, image, client_size):
    """Free nonbattle navigation; the destination still verifies inventory."""
    from types import SimpleNamespace

    from live_decision.inventory import recognize_inventory_button

    def recognize(node, picture, *, pipeline_override):
        config = dict(pipeline_override[node], threshold=.90)
        return context.run_recognition(node, picture, pipeline_override={node: config})
    return recognize_inventory_button(SimpleNamespace(run_recognition=recognize), image, client_size)


def read_continue_caption(rows, roi):
    """A decorative full-page continuation cue, after strong page identity."""
    return _row(rows, r'(?i:.*tap.*)|.*タップ.*', roi, minimum=.85)


def _common_caption_target(pages):
    boxes = [page.get('target') for page in pages]
    if any(not isinstance(box, (list, tuple)) or len(box) != 4
            or any(type(value) is not int for value in box) or min(box[2:]) <= 0 for box in boxes):
        raise ValueError('closing result caption geometry is incomplete')
    left, top = max(box[0] for box in boxes), max(box[1] for box in boxes)
    right, bottom = min(box[0] + box[2] for box in boxes), min(box[1] + box[3] for box in boxes)
    if right - left < 28 or bottom - top < 12:
        raise ValueError('closing result captions have no usable common interior')
    return (left + 2, top + 2, right - left - 4, bottom - top - 4)


class UnknownRecoveryPage(ValueError):
    """No supported nonbattle page was recognized in this frame."""


def _current_rows(context, image, *, date_probe=True, settings_probe=True):
    node = 'ProduceHifCurrentIntervalOCR'
    hit = context.run_recognition(node, image, pipeline_override={node: {
        'recognition': 'OCR', 'roi': [0, 0, 720, 1280], 'expected': '.+'}})
    rows = [dict(text=''.join(unicodedata.normalize('NFKC', row.text).split()), box=list(row.box), score=row.score)
        for row in hit.filtered_results] if hit and hit.hit else []
    if date_probe:
        from .nonbattle_date import detect_date
        detect_date(context, image, rows)
    if settings_probe:
        from .produce_settings import detect_produce_settings
        detect_produce_settings(context, rows)
    return rows


def current_inventory_summary_rows(context, image, *, frame_id):
    """Current list fields only; callers must use full OCR for pending details."""
    from .nonbattle_inventory import flat_inventory_count
    node = 'ProduceHifInventorySummaryOCR'
    regions = {'ProduceHifInventoryHeaderOCR': (10, 10, 510, 110),
        'ProduceHifInventoryFooterOCR': (65, 1010, 475, 240)}
    override = {name: dict(recognition='OCR', roi=list(roi), expected='.+') for name, roi in regions.items()}
    override[node] = dict(recognition='And', all_of=list(regions))
    detail = context.run_recognition(node, image, pipeline_override=override)
    if (detail is None or getattr(detail, 'hit', None) is not True
            or getattr(getattr(detail, 'algorithm', None), 'value', getattr(detail, 'algorithm', None)) != 'And'):
        raise ValueError('inventory summary lacks its complete current And result')
    groups = detail.all_results
    if not isinstance(groups, (list, tuple)) or len(groups) != 1:
        raise ValueError('inventory summary lacks its one complete group')
    children = groups[0].sub_results
    if not isinstance(children, (list, tuple)) or len(children) != len(regions):
        raise ValueError('inventory summary did not observe both current regions')
    rows, evidence = [], []
    for (name, roi), child in zip(regions.items(), children, strict=True):
        if (child.name != name or child.hit is not True
                or getattr(child.algorithm, 'value', child.algorithm) != 'OCR'):
            raise ValueError('inventory summary child identity or algorithm changed')
        observed = [dict(text=''.join(unicodedata.normalize('NFKC', row.text).split()),
            box=list(row.box), score=row.score) for row in child.filtered_results]
        rows.extend(observed)
        evidence.append(dict(name=name, roi=list(roi), reco_id=child.reco_id, rows=observed))
    # Compute regions may combine fields. Their original complete field ROIs,
    # confidence, unique labels and total-count bounds remain authoritative.
    if read_nonbattle_inventory(rows)['phase'] != 'inventory':
        raise ValueError('inventory summary is not the current permanent-card list')
    flat_inventory_count(rows)
    return rows, dict(source='same_frame_inventory_summary_ocr', frame_id=frame_id,
        reco_id=detail.reco_id, recognition_parameters=override, children=evidence)


def current_final_resource_rows(context, image, *, frame_id):
    """Fresh calendar/resources after the complete ordinary page was checked."""
    from .calendar import read_calendar, read_route_resources
    # Reuse the ordinary full-page detector scale. The former header crop
    # turned a clear day label into a weak result on the actual full11 frame.
    node, roi = 'ProduceHifFinalResourcesOCR', [0, 0, 720, 1280]
    config = dict(recognition='OCR', roi=roi, expected='.+')
    hit = context.run_recognition(node, image, pipeline_override={node: config})
    rows = [dict(text=''.join(unicodedata.normalize('NFKC', row.text).split()), box=list(row.box), score=row.score)
        for row in hit.filtered_results] if hit is not None and hit.hit else []
    evidence = dict(source='same_frame_final_resource_ocr', frame_id=frame_id,
        reco_id=getattr(hit, 'reco_id', None), recognition_parameters=config, rows=rows)
    try:
        if read_calendar(rows) != (3, 'outing'):
            raise ValueError('final resources are not the original ordinary third day')
        read_route_resources(rows)
    except ValueError as error:
        error.evidence = evidence
        raise
    return rows, evidence


def _interval_resources(rows):
    _text(rows, r'インターバル', (25, 15, 180, 60))
    stamina = _text(rows, r'\d{1,2}/\d{1,2}', (285, 35, 165, 75))
    currency = int(_text(rows, r'\d{1,6}', (350, 80, 110, 75)))
    current, maximum = map(int, stamina.split('/'))
    if not 0 <= current <= maximum <= 100 or maximum == 0:
        raise ValueError('contradictory HIF interval stamina')
    return current, maximum, currency


def read_interval_page(context, image, *, rows=None, menu=True):
    rows = _current_rows(context, image) if rows is None else rows
    current, maximum, currency = _interval_resources(rows)
    # The notebook keeps the interval HUD and replaces the shop controls with its tabs.
    if any(row['text'] in ('スケジュール', '育成目標', '編成') and row['score'] >= .95
           and 1070 <= row['box'][1] < 1150 for row in rows):
        raise ValueError('current interval notebook is open')
    return dict(phase='interval', target=interval_menu_target(image) if menu else None, stamina=current,
        max_stamina=maximum, currency=currency)


def read_interval_capacities(context, image, rows):
    """Read each complete current limit; a rejected full-page label proves none."""
    from live_decision.hif_recovery import _field
    counters, evidence = {}, {}
    for key, name, pattern, roi in (
            ('interval_upgrade_cards_remaining', 'IntervalUpgradeCapacity', r'強化可能枚数あと\d+枚', (75, 950, 225, 60)),
            ('interval_customizable_cards_remaining', 'IntervalCustomizationCapacity', r'カスタマイズ可能枚数あと\d+枚', (310, 950, 285, 60))):
        try:
            field = dict(_row(rows, pattern, roi), source='complete_full_page_label')
        except ValueError:
            field = dict(_field(context, image, name, roi, pattern), source='independent_complete_label')
        counters[key] = int(re.fullmatch(pattern.replace(r'\d+', r'(\d+)'), field['text'])[1])
        x, y, w, h = roi
        evidence[key] = dict(field, source_rows=[deepcopy(row) for row in rows
            if x <= row['box'][0] < x+w and y <= row['box'][1] < y+h])
    return counters, evidence


def read_interval_notebook(context, image, *, rows=None, frame_id=None, epoch=None):
    rows = _current_rows(context, image) if rows is None else rows
    current, maximum, currency = _interval_resources(rows)
    return _read_notebook_body(context, image, rows, dict(stamina=current, max_stamina=maximum, currency=currency),
        frame_id=frame_id, epoch=epoch)


def read_route_notebook(context, image, rows, *, frame_id=None, epoch=None):
    """Same notebook identity body under the ordinary day's current HUD."""
    from .daily import complete_daily_resources
    from .calendar import read_calendar, read_route_resources
    completed, _ = complete_daily_resources(context, image, rows)
    try:
        ordinary_day, _ = read_calendar(completed)
    except ValueError:
        ordinary_day = None
    return _read_notebook_body(context, image, completed, read_route_resources(completed), ordinary_day=ordinary_day,
        frame_id=frame_id, epoch=epoch)


def _ordinary_notebook_schedule(image, rows, day):
    """Ordinary countdown and selected timeline, independent of interval rounds."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('ordinary notebook selected-page pixels are not calibrated')
    selected = _row(rows, str(day)+r'日', (80, 335, 170, 730))
    x, y, w, h = selected['box']
    area = image[y-6:y+h+6, x-10:x+w+10]
    cyan = ((area[:, :, 0] >= 130) & (area[:, :, 1] >= 120) & (area[:, :, 2] <= 100)
        & (abs(area[:, :, 0].astype(int)-area[:, :, 1].astype(int)) <= 90))
    # The selected schedule has a thin white underline on its blue footer.
    footer = image[1137:1153, 40:248]
    white = np.min(footer, axis=2) >= 220
    blue = (footer[:, :, 0] >= 150) & (footer[:, :, 2] <= 100)
    lines = int(np.count_nonzero(np.sum(white, axis=1) >= 180))
    if float(np.mean(cyan)) < .70 or not 1 <= lines <= 6 or float(np.mean(blue)) < .75:
        raise ValueError('ordinary notebook schedule/day is not independently selected')
    return dict(source='ordinary_selected_schedule', calendar_day=day, selected_day=dict(selected),
        selected_day_cyan_fraction=float(np.mean(cyan)), schedule_indicator_roi=[40,1137,208,16],
        schedule_indicator_rows=lines, schedule_indicator_blue_fraction=float(np.mean(blue)))


def _read_notebook_body(context, image, rows, resources, *, ordinary_day=None, frame_id=None, epoch=None):
    from live_decision.hif_session import notebook_identity_fields
    from live_decision.hif_profiles import notebook_hif_identity
    from live_decision.hif_recovery import _field, _close_target

    current, maximum = resources['stamina'], resources['max_stamina']
    _text(rows, r'スケジュール', (50, 1070, 190, 80))
    _text(rows, r'育成目標', (280, 1070, 160, 80))
    formation = _row(rows, r'編成', (500, 1070, 150, 80))
    common = dict(resources)
    if ordinary_day is not None:
        common['day'] = ordinary_day
        try:
            evidence = _ordinary_notebook_schedule(image, rows, ordinary_day)
        except ValueError:
            pass
        else:
            return dict(common, phase='notebook', target=tuple(formation['box']), notebook_evidence=evidence)
    try:
        _text(rows, r'当日', (90, 610, 150, 110))
        _text(rows, r'ラウンド2', (390, 360, 210, 90))
        _text(rows, r'ラウンド1', (390, 760, 210, 100))
        return dict(common, phase='notebook', target=tuple(formation['box']))
    except ValueError:
        pass
    matches = []
    fields = {}
    def field(name, spec):
        key = (name, tuple(spec[0]), spec[1])
        if key not in fields:
            # Current full-image OCR can independently prove the complete title
            # inside its unchanged original ROI. Never expand a tight idol ROI
            # or promote a weak/partial row to avoid its original crop.
            value = None
            if name == 'title':
                normalized = [dict(row, text=''.join(unicodedata.normalize('NFKC', row['text']).split()),
                    source_text=row['text']) for row in rows]
                try:
                    value = dict(_row(normalized, spec[1], spec[0]), source='current_full_frame_ocr',
                        source_frame_id=frame_id, roi=list(spec[0]))
                except ValueError:
                    pass
            fields[key] = (value if value is not None else
                _field(context, image, name, *spec, frame_id=frame_id, epoch=epoch))
        return fields[key]
    for layout in ('legacy', 'campaign_banner'):
        specs = notebook_identity_fields(2, layout=layout)
        try:
            values = {name: field(name, specs[name])
                for name in ('title', 'song', 'idol')}
            profile = notebook_hif_identity(values['song']['text'], values['idol']['text'])
            matches.append((values, profile))
        except ValueError:
            continue
    if len(matches) != 1:
        raise ValueError('current interval notebook identity layout is unresolved')
    values, profile = matches[0]
    return dict(common, phase='identity', target=_close_target(context, image),
        identity=(profile.identity, profile.plan, current, maximum), rows=list(values.values()))


def read_nonbattle_inventory(rows):
    title = _row(rows, r'所持スキルカード|所持Pアイテム', (10, 10, 510, 110))
    if title['text'] == '所持スキルカード':
        from .nonbattle_observation import _within
        close = next((row for row in rows if row['text'] == '閉じる'
            and _within(row['box'], (180, 1090, 360, 160))), None)
        target = tuple(close['box']) if close is not None else (360, 1160, 1, 1)
        return dict(phase='inventory', target=target, rows=rows)
    close = _row(rows, r'閉じる', (180, 1090, 360, 160))
    return dict(phase='owned_items', target=tuple(close['box']), rows=rows)


def read_nonbattle_drink_controls(context, image, *, footer_offset=13):
    from live_decision.drink_panel import drink_slot_targets
    # Interval is 13px below Live; the ordinary first-day footer in 157 is
    # at the original Live coordinates. The caller selects its known layout,
    # never whichever offset happens to produce a positive result.
    if type(footer_offset) is not int or footer_offset not in (0, 13):
        raise ValueError('unsupported HIF empty-drink footer layout')
    aligned = np.zeros_like(image)
    aligned[1154:1244, :365] = image[1154 + footer_offset:1244 + footer_offset, :365]
    filled = drink_slot_targets(context, aligned)
    return filled, _nonbattle_empty_drink_controls(context, aligned)


def _nonbattle_empty_drink_controls(context, aligned):
    from live_decision.drink_panel import empty_drink_slot_targets

    path = resolve_runtime_path(Path('assets/resource/base/image/produce/hif_empty_drink_glyph.png'),
        Path('resource/base/image/produce/hif_empty_drink_glyph.png'))
    template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    size = tuple(context.gkh_hif_client_size)
    # A fixed raster profile for the verified 480px client family, selected
    # by native geometry, never by whichever contrast/offset passes a cup.
    profile = ('compact_route_opacity_binomial3' if 479 <= size[0] <= 481 and 852 <= size[1] <= 854 else 'standard')
    return empty_drink_slot_targets(aligned, template, client_size=size, render_profile=profile)


def read_interval_empty_drinks(context, image, *, footer_offset=13):
    filled, empty = read_nonbattle_drink_controls(context, image, footer_offset=footer_offset)
    if any(occupied is not None or vacant is None for occupied, vacant in zip(filled, empty, strict=True)):
        error = ValueError('current interval does not prove four independently empty drink slots')
        error.evidence = dict(footer_offset=footer_offset, occupied=list(filled), empty=list(empty))
        raise error
    return [dict(slot=index, state='EMPTY', box=[row['box'][0], row['box'][1] + footer_offset, *row['box'][2:]],
        confidence=row['confidence'], acceptance_threshold=row['acceptance_threshold'],
        glyph_evidence={key: value for key, value in row.items() if key != 'box'}) for index, row in enumerate(empty)]


def read_bound_drink_slots(context, image, drinks, *, footer_offset, maintained_revision=None):
    if maintained_revision is not None:
        return read_quoted_nonbattle_drinks(context, image, drinks, footer_offset=footer_offset,
            revision=maintained_revision, expected_count=len(drinks))
    if not drinks:
        return read_interval_empty_drinks(context, image, footer_offset=footer_offset)
    filled, empty = read_nonbattle_drink_controls(context, image, footer_offset=footer_offset)
    known = {drink.slot: drink for drink in drinks}
    if len(known) != len(drinks):
        raise ValueError('bound nonbattle drinks have duplicate physical slots')
    result = []
    for slot, (occupied, vacant) in enumerate(zip(filled, empty, strict=True)):
        if slot in known:
            if occupied is None or vacant is not None:
                raise ValueError('current occupied drink control differs from its bound physical slot')
            drink = known[slot]
            result.append(dict(slot=slot, state='OCCUPIED', instance_id=drink.instance_id, drink_id=drink.drink_id,
                identity_source='maintained_bound_instance', box=[occupied[0], occupied[1] + footer_offset, *occupied[2:]]))
        else:
            if occupied is not None or vacant is None:
                raise ValueError('current empty drink control differs from its bound physical slot')
            result.append(dict(slot=slot, state='EMPTY', box=[vacant['box'][0], vacant['box'][1] + footer_offset, *vacant['box'][2:]],
                confidence=vacant['confidence'], acceptance_threshold=vacant['acceptance_threshold'],
                glyph_evidence={key: value for key, value in vacant.items() if key != 'box'}))
    return result


def read_current_drink_occupancy(context, image, *, footer_offset):
    """Read all physical controls without guessing identities of new gifts."""
    filled, empty = read_nonbattle_drink_controls(context, image, footer_offset=footer_offset)
    result = []
    for slot, (occupied, vacant) in enumerate(zip(filled, empty, strict=True)):
        if (occupied is None) == (vacant is None):
            raise ValueError('current drink slot has contradictory or unknown occupancy')
        if occupied is not None:
            result.append(dict(slot=slot, state='OCCUPIED', box=[occupied[0], occupied[1] + footer_offset, *occupied[2:]]))
        else:
            result.append(dict(slot=slot, state='EMPTY', box=[vacant['box'][0], vacant['box'][1] + footer_offset, *vacant['box'][2:]],
                confidence=vacant['confidence'], acceptance_threshold=vacant['acceptance_threshold'],
                glyph_evidence={key: value for key, value in vacant.items() if key != 'box'}))
    return result


def nonbattle_drink_slots_consistent(slots):
    """The current route page reports the physical state of slots 0 through 3."""
    return (isinstance(slots, (list, tuple)) and len(slots) == 4
        and all(isinstance(row, dict) and type(row.get('slot')) is int and row['slot'] == index
            and row.get('state') in ('EMPTY', 'OCCUPIED') for index, row in enumerate(slots)))


def maintain_empty_nonbattle_drinks(context, image, drinks, *, footer_offset, revision):
    """Retain an unchanged empty ledger during ordinary, nonconsuming UI.

    This is not a fresh occupancy census and must not settle drink rewards,
    discards or consumption. Known page/resources and the unchanged ledger
    are checked by the caller. An occupied anchor or upstream bottle signal
    contradicts that retained state; neither supplies a new drink identity.
    """
    from live_decision.drink_panel import drink_slot_targets

    if drinks or type(revision) is not int or revision < 0:
        raise ValueError('ordinary empty-drink maintenance requires its empty ledger revision')
    if getattr(image, 'shape', ()) != (1280, 720, 3) or type(footer_offset) is not int or footer_offset not in (0, 13):
        raise ValueError('ordinary empty-drink maintenance layout is uncalibrated')
    aligned = np.zeros_like(image)
    aligned[1154:1244, :365] = image[1154 + footer_offset:1244 + footer_offset, :365]
    filled = drink_slot_targets(context, aligned)
    result = []
    # miyu1019/MaaGakumasu a8e46085, drinks.py:_slot_has_drink:
    # sample the inner dish every 2px; channel spread >80 in >10% is a
    # bottle signal. Its ordinary footer is the +13px layout (cy=1210).
    cy = 1197 + footer_offset
    for slot, cx in enumerate((62, 146, 230, 314)):
        pixels = image[cy-20:cy+21:2, cx-20:cx+21:2]
        fraction = float(np.mean(pixels.max(axis=2).astype(int)-pixels.min(axis=2) > 80))
        if filled[slot] is not None or fraction > .10:
            error = ValueError('ordinary drink controls contradict the maintained empty ledger')
            error.evidence = dict(slot=slot, occupied_anchor=filled[slot], colored_fraction=fraction,
                maintained_from_revision=revision, footer_offset=footer_offset)
            raise error
        result.append(dict(slot=slot, state='EMPTY', box=[cx-20, cy-20, 41, 41],
            occupancy_source='maintained_empty_ledger', maintained_from_revision=revision,
            colored_fraction=fraction))
    return result


def _maintained_nonbattle_bottle_presence(context, aligned):
    """Native per-slot presence only; not a calibrated drink input target."""
    from live_decision.layout import calibrated_live_client_size

    small = calibrated_live_client_size(context.gkh_hif_client_size) == (480, 853)
    slots = []
    for index, x in enumerate((17, 102, 187, 273)):
        node = f'LiveStatusDrinkSlot{index}'
        roi = (x, 1154, 90, 90)
        hit = context.run_recognition(node, aligned, pipeline_override={node: dict(
            recognition='TemplateMatch', roi=list(roi),
            template=[f'produce/hif_live_drink_{i}_480.png' if small else f'produce/drink_{i}.png' for i in (1, 2, 3)],
            green_mask=True, threshold=.8)})
        found = list(hit.filtered_results) if hit and hit.hit else []
        boxes = [tuple(row.box) for row in found]
        if any(len(box) != 4 or any(type(value) is not int for value in box)
                or box[2:] != (70, 70) or not x <= box[0] <= x + 20
                or not 1154 <= box[1] <= 1174 for box in boxes):
            raise ValueError('ordinary bottle presence is outside its complete physical slot')
        if len(boxes) > 1:
            raise ValueError('ordinary bottle presence is ambiguous')
        slots.append(boxes[0] if boxes else None)
    return tuple(slots)


def read_quoted_nonbattle_drinks(context, image, drinks, *, footer_offset, revision, expected_count):
    """Observe bottles and reconcile the original unchanged/awarded count.

    Empty placeholders do not add identity evidence. The upstream inner-dish
    signal may locate a bottle when its ring template misses. Physical
    details still need to prove every new bottle's identity before settlement.
    New bottle identities are deliberately absent and need physical details.
    """
    from live_decision.drink_panel import drink_slot_targets

    known = {drink.slot: drink for drink in drinks}
    if (len(known) != len(drinks) or any(type(slot) is not int or slot not in range(4) for slot in known)
            or type(expected_count) is not int or not len(known) <= expected_count <= 4
            or type(revision) is not int or revision < 0):
        raise ValueError('ordinary drink count lacks its original ledger or reward quote')
    if getattr(image, 'shape', ()) != (1280, 720, 3) or type(footer_offset) is not int or footer_offset not in (0, 13):
        raise ValueError('ordinary quoted-drink layout is uncalibrated')
    aligned = np.zeros_like(image)
    aligned[1154:1244, :365] = image[1154 + footer_offset:1244 + footer_offset, :365]
    # This path can only retain already identified bottles during ordinary
    # nonconsuming navigation. Its native slot match is presence evidence,
    # not the Live control registration used to open/use/discard a bottle.
    maintained_presence = bool(known) and expected_count == len(known)
    filled = (_maintained_nonbattle_bottle_presence(context, aligned) if maintained_presence
        else drink_slot_targets(context, aligned))
    empty, excluded = None, {}
    if maintained_presence and any(box is not None and slot not in known for slot, box in enumerate(filled)):
        # A raw ring candidate is not a registered Live control. A complete
        # current cup positively excludes that candidate during maintenance;
        # missing cups never turn unexplained candidates into empty slots.
        empty = _nonbattle_empty_drink_controls(context, aligned)
        filled = list(filled)
        for slot, box in enumerate(filled):
            if box is not None and slot not in known and empty[slot] is not None:
                excluded[slot] = dict(candidate_box=list(box), empty_glyph=empty[slot])
                filled[slot] = None
    cy = 1197 + footer_offset
    colored = []
    for cx in (62, 146, 230, 314):
        pixels = image[cy-20:cy+21:2, cx-20:cx+21:2]
        colored.append(float(np.mean(pixels.max(axis=2).astype(int)-pixels.min(axis=2) > 80)))
    upstream_presence = set()
    filled = list(filled)
    for slot, cx in enumerate((62, 146, 230, 314)):
        if filled[slot] is None and colored[slot] > .10:
            if empty is None:
                empty = _nonbattle_empty_drink_controls(context, aligned)
            if empty[slot] is None:
                # miyu drinks.py:_slot_has_drink uses this same inner dish.
                # This target only opens the actual slot's normal detail;
                # it supplies no drink identity or permission to consume.
                filled[slot] = (cx-20, 1197-20, 41, 41)
                upstream_presence.add(slot)
    occupied = {slot for slot, box in enumerate(filled) if box is not None}
    if len(occupied) != expected_count or not set(known) <= occupied:
        error = ValueError('current bottles differ from the original quoted physical count')
        error.evidence = dict(occupied_slots=sorted(occupied), held_slots=sorted(known), expected_count=expected_count)
        raise error
    result = []
    for slot, cx in enumerate((62, 146, 230, 314)):
        # Same a8e46085 inner-dish signal as the empty-ledger maintenance.
        fraction = colored[slot]
        if (filled[slot] is not None) != (fraction > .10):
            # A colored dialogue background can shine through a known empty
            # cup (retained lesson 027). Use the existing positive glyph only
            # to resolve that actual contradiction, never to audit every empty
            # control. It also rejects an anchor on a positively empty cup.
            if empty is None:
                empty = _nonbattle_empty_drink_controls(context, aligned)
            if filled[slot] is None and empty[slot] is None:
                raise ValueError('colored drink control has no current physical bottle anchor')
            if filled[slot] is not None and empty[slot] is not None:
                raise ValueError('current bottle anchor contradicts a positively observed empty cup')
        row = dict(slot=slot, state='OCCUPIED' if slot in occupied else 'EMPTY',
            box=[filled[slot][0], filled[slot][1]+footer_offset, *filled[slot][2:]] if slot in occupied else [cx-20, cy-20, 41, 41],
            occupancy_source=('observed_nonbattle_slot_presence' if maintained_presence else 'observed_occupied_anchor')
                if slot in occupied else 'maintained_quoted_count',
            maintained_from_revision=revision, expected_count=expected_count, colored_fraction=fraction)
        if slot in known:
            row.update(instance_id=known[slot].instance_id, drink_id=known[slot].drink_id,
                identity_source='maintained_bound_instance')
        if slot in upstream_presence:
            row['occupancy_source'] = 'upstream_inner_dish_color_requires_detail'
        if slot in excluded:
            row['excluded_unregistered_ring'] = excluded[slot]
        result.append(row)
    return result


class HifRecoveryProducer:
    def __init__(self, context, *, run_id, game_pid, clock, evidence_sink=None,
            recovery_report=None, terminal_report=None, ap_clock_reports=None, terminal_runtime=None,
            capture_guard=None, capture_record_guard=None, resume_probe_home=False):
        if capture_guard is not None and not callable(capture_guard):
            raise ValueError('HIF route capture guard must be a trusted callable')
        if type(resume_probe_home) is not bool or resume_probe_home and capture_guard is None:
            raise ValueError('HIF same-save probe home requires its trusted capture guard')
        self.context, self.run_id, self.game_pid, self.clock = context, run_id, game_pid, clock
        self.capture_guard = capture_guard
        if capture_record_guard is not None and not callable(capture_record_guard):
            raise ValueError('HIF route record capture guard must be a trusted callable')
        self.capture_record_guard = capture_record_guard
        self.resume_probe_home = resume_probe_home
        self._external_evidence_sink = evidence_sink
        self._source_frames = OrderedDict()
        self._frozen_source_frames = {}
        self._source_capture_store = self._source_capture_session = None
        self.evidence_sink = self._remember_evidence
        self.host = None
        self.identity = None
        self.identity_frames = ()
        self.rows = []
        self.events = []
        self.recovery_report = recovery_report
        self.terminal_report = terminal_report
        self.terminal_runtime, self.terminal_remainder, self._remainder_scope = terminal_runtime, None, None
        self._recovery_scope = None
        self._round_one_entry = None
        self._round_one_entry_scope = None
        self.ap_clock_reports = ap_clock_reports
        self.ap_readback_report = None
        self.last_interval = None
        self.last_notebook_day = None
        self.inventory = None
        self.inventory_scope = None
        self.inventory_navigation = None
        self.complete_ledger = None
        self.current_frame_id = None
        self.last_frame_ids = ()
        self.supply = None
        self.daily = None
        self.ending = False
        self.preparation_page = None

    def _remember_evidence(self, image, metadata):
        key = metadata.get('frame_id')
        if isinstance(key, str) and getattr(image, 'shape', ()) == (1280, 720, 3):
            self._source_frames[key] = (image.copy(), deepcopy(metadata))
            while len(self._source_frames) > 64:
                self._source_frames.popitem(last=False)
        if self._external_evidence_sink is not None:
            self._external_evidence_sink(image, metadata)

    def freeze_input_source(self, page):
        """Bind the original registration. Full pixels are an optional local capture."""
        keys = page.frame_ids
        if not source_frames_match(page.observation, keys) or any(key not in self._source_frames for key in keys):
            raise ValueError('route dispatch lacks its original registered source images')
        key = keys[0]
        if re.fullmatch(r'route-frame-' + str(self.game_pid) + r'-([0-9]+)', key) is None:
            raise ValueError('route source frame identity is not this game capture')
        image, metadata = self._source_frames[key]
        if getattr(image, 'shape', ()) != (1280, 720, 3):
            raise ValueError('route source image is outside its calibrated viewport')
        record = dict(metadata, frame_id=key)
        record.pop('path', None)
        store = get_runtime_store()
        if not store.config.enabled or store.config.mode != 'screenshot':
            return [deepcopy(record)]
        if self._source_capture_store is not store:
            self._source_capture_store, self._source_capture_session = store, None
            self._frozen_source_frames.clear()
        stored = self._frozen_source_frames.get(key)
        if stored is not None and Path(stored['path']).is_file():
            return [deepcopy(stored)]
        success, pixels = cv2.imencode('.png', image)
        if not success:
            raise ValueError('route source image encoding failed')
        try:
            if self._source_capture_session is None:
                self._source_capture_session = store.start_session(_ROUTE_SCREENSHOT_ROLE)
            path = store.write_image('normal', self._source_capture_session, pixels.tobytes(),
                metadata=self._route_screenshot_metadata(record), capture_kind='screenshot', suffix='.png')
        except LocalDataError as error:
            logger.warning(f'HIF route screenshot could not be retained: {error}')
            return [deepcopy(record)]
        self._frozen_source_frames = {k: v for k, v in self._frozen_source_frames.items() if k in self._source_frames}
        self._frozen_source_frames[key] = dict(record, path=str(path.resolve()))
        return [deepcopy(self._frozen_source_frames[key])]

    def _route_screenshot_metadata(self, record):
        return dict(image_role=_ROUTE_SCREENSHOT_ROLE, run_id=self.run_id, game_pid=self.game_pid,
            frame_id=record['frame_id'], phase=record.get('phase'))

    def original_route_image_path(self, record):
        """Resolve an existing route PNG bound by the receipt or its capture manifest."""
        source = record.get('path')
        if not isinstance(source, str):
            raise ValueError('original route pixels were not retained')
        path = Path(source).resolve()
        if not path.is_file():
            raise ValueError('original route pixels are unavailable')
        if (path.is_relative_to((self.host.path.parent / 'input-frames').resolve())
                and path.name == record['frame_id'] + '.png'):
            return path
        root = (get_runtime_store().config.root / 'images' / 'normal').resolve()
        if path.suffix != '.png' or not path.is_relative_to(root):
            raise ValueError('original route pixels are outside approved local evidence')
        try:
            manifest = json.loads(path.with_suffix('.png.json').read_bytes())
        except (OSError, ValueError) as error:
            raise ValueError('original route screenshot manifest is unavailable') from error
        if not isinstance(manifest, dict) or (manifest.get('capture_kind'), manifest.get('image_file'),
                manifest.get('session_id'), manifest.get('metadata')) != (
                'screenshot', path.name, path.parent.name, self._route_screenshot_metadata(record)):
            raise ValueError('original route screenshot manifest does not bind this frame')
        return path

    @property
    def input_events(self):
        return self.host.input_events if self.host is not None else []

    def current_inventory(self):
        if self.inventory is not None:
            return self.inventory.record()
        from .recovery_inventory import retained
        return retained(self.host, complete=True)

    def prepare_inventory_census(self):
        if self.host is None or self.host.ownership['phase'] != 'recovery':
            return None
        from .recovery_inventory import scope, retained
        from .nonbattle_inventory import bind_inventory_census
        return bind_inventory_census(self, ['recovery', scope(self.host)], retained(self.host))

    def prepare_recovery_epoch(self):
        """A warm process cannot promote the prior round's census to current."""
        if self.host is None or (self.host.ownership or {}).get('phase') != 'recovery':
            return
        from .recovery_inventory import scope
        current = scope(self.host)
        if current == self._recovery_scope: return
        if self._recovery_scope is not None:
            self.recovery_report = None
        self.identity, self.identity_frames = None, ()
        self.inventory = self.inventory_navigation = self.complete_ledger = self.last_interval = None
        self.inventory_scope = None
        self.terminal_remainder, self._remainder_scope = None, None
        self.ending = False
        self._recovery_scope = current

    def prepare_drink_remainder(self):
        from .recovery_inventory import scope
        if (self.host.ownership['phase'] != 'recovery' or self.host.ledger is None or not self.host.ledger.state.drinks
                or (self.host.handoff or {}).get('terminal_receipt', {}).get('round_number') != 1):
            return
        current = scope(self.host)
        if self._remainder_scope == current: return
        if self.terminal_runtime is None:
            raise ValueError('post-battle drink recovery lacks its trusted terminal runtime')
        try:
            from live_decision.hif_terminal_drinks import load_verified_terminal_drink_remainder
        except ImportError as error:
            raise ValueError('verified terminal drink remainder reader is not installed') from error
        record = json.loads(self.host.path.read_text(encoding='utf-8'))
        self.terminal_remainder = load_verified_terminal_drink_remainder(self.terminal_runtime, record, report=self.terminal_report)
        self._remainder_scope = current

    def _page(self, image, *, rows=None):
        from live_decision.hif_recovery import read_recovery_page
        closing_error = None
        if rows is None:
            rows = _current_rows(self.context, image)
        else:
            from .nonbattle_date import detect_date
            from .produce_settings import detect_produce_settings
            detect_date(self.context, image, rows)
            detect_produce_settings(self.context, rows)
        self.rows = rows
        from .opening_memory import observe_opening_memory
        observe_opening_memory(self, rows)
        if (self.host is not None and self.identity is None
                and (getattr(self.host, 'ownership', None) or {}).get('phase') == 'recovery'):
            from .recovery_inventory import retained_identity
            saved_identity = retained_identity(self.host)
            if saved_identity is not None: self.identity, self.identity_frames = saved_identity
        if (self.host is not None and (self.host.handoff or {}).get('terminal_receipt', {}).get('round_number') == 2):
            from .ending import read_ending_page
            page = read_ending_page(self.context, image, rows, self.host, terminal_report=self.terminal_report)
            self.ending = True
            return page
        if (self.host is not None and self.host.ownership['phase'] == 'recovery'
                and (self.host.handoff or {}).get('terminal_receipt', {}).get('round_number') == 1
                and any(row['score'] >= .95 and 'ラウンド1結果' in row['text'] for row in rows)):
            from .ending import verified_ending_receipt, read_verified_round_one_result
            from .hif import active_hif_role
            receipt = verified_ending_receipt(self.host, terminal_report=self.terminal_report, round_number=1)
            return read_verified_round_one_result(rows, receipt, role=active_hif_role(self.host))
        if (self.host is not None and self.host.ownership['phase'] == 'recovery'
                and (self.host.handoff or {}).get('terminal_receipt', {}).get('round_number') == 1):
            from .ending import read_ending_page
            try:
                page = read_ending_page(self.context, image, rows, self.host,
                    terminal_report=self.terminal_report, round_number=1)
            except ValueError as error:
                # The reward can lead back to the route or to actual closing
                # pages. Neither the score nor the reward predicts which one.
                self.ending = False
                closing_error = str(error)
            else:
                self.ending = True
                return page
        if self.host is not None and any(row['text'] == 'Pドリンク詳細' for row in rows):
            from .recovery_drinks import request, read_detail
            from .activity_production import drink_catalog
            value = request(self.host)
            if value is None or value.get('active_slot') not in range(4):
                raise ValueError('cold drink detail has no original same-epoch physical slot')
            detail = read_detail(self.context, image, rows, drink_catalog())
            return dict(phase='recovery_drink_detail', target=detail['target'], slot=value['active_slot'],
                drink_id=detail['drink_id'], drink_detail_evidence=detail)
        if self.host is not None and self.host.ledger is None:
            from .new_run import start_record
            state = start_record(self.host.handoff)
            if state is not None and state.get('status') in ('reserved', 'created_ap_unverified', 'complete'):
                from .start_transaction import read_start_created
                if state['status'] == 'complete':
                    from .home import read_active_home
                    from .date_recovery import read_saved_resume
                    try:
                        resumed = read_saved_resume(image, rows, saved_day=1, nonbattle=True)
                    except ValueError:
                        pass
                    else:
                        return dict(phase='start_resume_confirm', target=resumed['target'])
                    try:
                        home = read_active_home(image, rows, preview_saved=True, optional_preview_ap=True)
                    except ValueError:
                        pass
                    else:
                        return dict(phase='start_resume_home', target=home['target'],
                            ap=home['ap'], max_ap=home['max_ap'], home_evidence=home)
                if state['status'] == 'created_ap_unverified' and any(
                        row['score'] >= .85 and row['text'] == '保存して中断' for row in rows):
                    from .start_transaction import read_start_ap_menu
                    return read_start_ap_menu(rows)
                if state['status'] == 'created_ap_unverified' and any(
                        row['score'] >= .85 and row['text'] == 'ホーム' for row in rows):
                    from .start_transaction import read_start_ap_home
                    return read_start_ap_home(image, rows, context=self.context, read_clock=self.ap_clock_reports is not None)
                if any(row['score'] >= .85 and row['text'] == 'H.I.F本戦まで' for row in rows):
                    if state['status'] == 'complete':
                        from .first_day import read_first_day_page
                        return read_first_day_page(self, image, rows)
                    return read_start_created(rows, image=image,
                        enable_ap_navigation=state['status'] == 'created_ap_unverified')
            from .prestart import read_prestart_page
            page = read_prestart_page(self.context, image, rows, host=self.host)
            if page is not None:
                return page
        try:
            page = read_nonbattle_inventory(rows)
        except ValueError:
            page = None
        if page is not None:
            from .first_day import uses_initial_deck_observation
            if page['phase'] == 'inventory' and uses_initial_deck_observation(self.host):
                return page
            saved = self.prepare_inventory_census()
            if page['phase'] == 'inventory' and self.inventory is None:
                from .nonbattle_inventory import FlatInventoryCollector, load_flat_inventory_readers
                self.inventory = FlatInventoryCollector(*load_flat_inventory_readers())
                self.inventory.census_session = self.inventory_scope
                if saved is not None:
                    self._restore_inventory(retained=saved)
            return page
        try:
            page = read_interval_page(self.context, image, rows=rows, menu=self.identity is None)
            if self.identity is not None and (self.current_inventory() or {}).get('complete') is True:
                page['drink_slots'] = read_current_drink_occupancy(self.context, image, footer_offset=13)
                from .recovery_drinks import next_slot
                proof = dict(kind='root', slots=page['drink_slots'], **{key: page[key] for key in ('stamina', 'max_stamina', 'currency')})
                slot = next_slot(self.host, proof)
                if slot is not None:
                    page.update(drink_inspect_slot=slot, target=tuple(page['drink_slots'][slot]['box']))
                page['counters'], page['capacity_evidence'] = read_interval_capacities(self.context, image, rows)
            elif self.identity is not None:
                matches, _, _, _ = recognize_route_inventory_button(self.context, image,
                    tuple(self.context.gkh_hif_client_size))
                if len(matches) != 1:
                    raise ValueError('current nonbattle inventory control is not uniquely observed')
                page['target'] = tuple(matches[0].box)
                page['inventory_navigation'] = True
            return page
        except ValueError as interval_error:
            try:
                return read_interval_notebook(self.context, image, rows=rows,
                    frame_id=self.current_frame_id, epoch=getattr(self.clock, 'epoch', None))
            except ValueError as notebook_error:
                try:
                    return read_recovery_page(self.context, image, round_number=2,
                        frame_id=self.current_frame_id, epoch=getattr(self.clock, 'epoch', None))
                except ValueError as error:
                    raise UnknownRecoveryPage(f'unknown nonbattle recovery page: {interval_error}; {notebook_error}; {error}'
                        + (f'; round-one reward/closing: {closing_error}' if closing_error else '')) from error

    def _restore_inventory(self, *, collector=None, retained=None, phase='inventory'):
        if self.host is None:
            return
        gate = self.host.hooks.transition_gate
        retained = (self.host.handoff or {}).get('nonbattle_inventory') if retained is None else retained
        inventory = self.inventory if collector is None else collector
        if not retained or (gate.pending is not None and gate.pending.target.kind not in (
                'recovery_inventory_detail', 'recovery_inventory_scroll')):
            return
        durable = retained.get('durable_frames') or ()
        if phase == 'reward_inventory' and not durable:
            raise ValueError('course inventory navigation lacks its durable current registration')
        if self.recovery_report is None and not durable:
            return
        inventory.slots = {row['index']: deepcopy(row) for row in retained['slots']}
        inventory.total = retained['total']
        inventory.frame_ids = list(retained.get('frame_ids', ()))
        inventory.registration = deepcopy(retained.get('registration'))
        frame_records = {row['frame_id']: row for row in durable}
        report_frames = ({row['frame_id']: row for row in (self.recovery_report or {}).get('frames', ())}
            if phase != 'reward_inventory' else {})
        battle_images = ((Path(self.terminal_runtime) / '.local/runtime-data/images/normal').resolve()
            if self.terminal_runtime is not None else None)

        def load_frame(key):
            record = frame_records.get(key) or report_frames.get(key)
            if record is None:
                raise ValueError('retained inventory source frame is missing from its original report')
            current = self._source_frames.get(key)
            if current is not None and record.get('phase') == current[1].get('phase'):
                return current[0]
            path = Path(record['path']).resolve() if isinstance(record.get('path'), str) else None
            if not (path is not None and key not in frame_records and battle_images is not None
                    and path.is_relative_to(battle_images)):
                path = self.original_route_image_path(record)
            image = cv2.imread(str(path))
            if image is None or image.shape != (1280, 720, 3):
                raise ValueError('retained inventory source image is unavailable')
            return image
        if phase == 'reward_inventory':
            inventory.selected_index = retained['selected_index']
        else:
            inventory.revalidate_retained_details(self.context, load_frame)
            details = [(int(row['detail']['frame_ids'][-1].rsplit('-', 1)[1]), row['index'])
                for row in inventory.slots.values() if row.get('detail')]
            inventory.selected_index = retained.get('selected_index', max(details)[1] if details else None)
        if phase == 'reward_inventory' or gate.pending is None or retained.get('first_day_once') and durable:
            keys = retained.get('durable_frame_ids')
            if (not isinstance(keys, list) or len(keys) != 1
                    or len(set(keys)) != len(keys) or [row.get('frame_id') for row in durable] != keys):
                raise ValueError('closed inventory navigation lacks its durable current registration')
            if phase == 'reward_inventory' and gate.pending is not None and (
                    keys != retained.get('review_frame_ids')
                    or int(keys[0].rsplit('-', 1)[1]) + gate.pending.source.capture_age_ms != gate.pending.source.revision):
                raise ValueError('course inventory image is not the original navigation source')
            boxes = retained.get('visible_boxes') or {}
            source_images = [load_frame(key) for key in keys]
            from .nonbattle_inventory import flat_inventory_boxes, align_flat_inventory_rows
            expected = align_flat_inventory_rows(source_images, flat_inventory_boxes(retained['total'], retained['displacement']))
            if (len(boxes) < 2 or any(not key.isdigit() or int(key) not in expected
                    or not isinstance(box, (list, tuple)) or len(box) != 4
                    or any(type(v) is not int for v in box)
                    or tuple(box) != tuple(expected[int(key)]) for key, box in boxes.items())
                    or len({box[0] for box in boxes.values()}) < 2):
                raise ValueError('closed inventory registration lacks its original physical columns')
            inventory.previous = source_images[len(keys) // 2]
            inventory.previous_frame_ids = tuple(keys)
            inventory.previous_boxes = {int(key): tuple(box) for key, box in boxes.items()}
            inventory.displacement = retained['displacement']
            return
        if gate.pending.target.kind in ('recovery_inventory_scroll', 'recovery_inventory_detail'):
            if self.recovery_report is None:
                raise ValueError('pending inventory navigation needs its successful original dispatch report')
            matches = [event for event in self.recovery_report['route']['events']
                if event.get('phase') == phase and len(event.get('frame_ids', ())) == 1 and source_frames_match(gate.pending.source, event.get('frame_ids', ()))
                and int(event['frame_ids'][-1].rsplit('-', 1)[1]) + event['capture_age_ms'] == gate.pending.source.revision]
            if len(matches) != 1:
                raise ValueError('retained inventory navigation source does not match its original pending')
            keys = matches[0]['frame_ids']
            inventory.previous = load_frame(keys[len(keys) // 2])
            inventory.previous_frame_ids = tuple(keys)
            inventory.previous_boxes = {row['index']: tuple(row['box']) for row in retained['slots']
                if row['frame_ids'] == keys}
            if len(inventory.previous_boxes) < 2:
                raise ValueError('retained inventory navigation lacks its current visible thumbnail anchors')
            inventory.displacement = retained['displacement']

    def observation_window(self, record, now):
        """The first returned R1 entry has its own upstream read-only recovery."""
        from .host import RECOVERY_NAVIGATION
        owner, handoff = record['ownership'], record.get('handoff') or {}
        scope = (record['run_id'], owner['revision'])
        terminal = handoff.get('terminal_receipt') or {}
        self._round_one_entry = None
        if (owner['owner'] != 'task600' or owner['phase'] != 'recovery' or terminal.get('round_number') != 1
                or not terminal.get('accepted') or not terminal.get('terminal') or scope == self._round_one_entry_scope
                or record['transition'].get('pending') is not None or record['transition'].get('stopped')
                or (record.get('ledger') or {}).get('pending') is not None):
            return None
        # A cold launch part-way through the reward/inventory flow still has
        # the R1 terminal. Its last navigation records whether entry has passed.
        carrier = (handoff.get('nonbattle_context') or {}).get('route_input_receipt') or {}
        original = (carrier.get('original_transition') or {}).get('pending') or {}
        if (handoff.get('recovery_navigation_receipts') or handoff.get('nonbattle_inventory')
                or (original.get('target') or {}).get('kind') in RECOVERY_NAVIGATION
                and (original.get('source') or {}).get('page_id', '').startswith('recovery:')):
            return None
        # fb8 ProduceEntryHIF (120s), followed on timeout by the first
        # EntryRecoveryAuto observation (30s). This covers only this entry.
        self._round_one_entry = (now + 120_000, now + 150_000)
        self._round_one_entry_scope = scope
        return 150_000

    def observe(self, *, reconciliation=False, _deadline=None, _deck_settle_deadline=None):
        self.prepare_recovery_epoch()
        self.prepare_inventory_census()
        from .customization_transaction import request as customization_request
        if self.host is not None and customization_request(self.host) is not None:
            if getattr(self, 'customization_producer', None) is None:
                from .customization_production import HifCustomizationProducer
                self.customization_producer = HifCustomizationProducer(self)
            return self.customization_producer.observe(reconciliation=reconciliation)
        from .rewards import active
        if self.host is not None and active(self.host):
            if getattr(self, 'reward_producer', None) is None:
                from .reward_production import HifRewardProducer
                self.reward_producer = HifRewardProducer(self)
            return self.reward_producer.observe()
        from .activity_reward import request as activity_request
        if self.host is not None and (activity_request(self.host) is not None or
                self.host.ownership['phase'] == 'nonbattle' and self.host.ledger.state.day in (3, 6)):
            if getattr(self, 'activity_producer', None) is None:
                from .activity_production import HifActivityProducer
                self.activity_producer = HifActivityProducer(self)
            return self.activity_producer.observe(reconciliation=reconciliation)
        if self.host is not None and self.host.ownership['phase'] == 'nonbattle':
            if self.host.ledger.state.day < 7:
                if self.daily is None:
                    from .daily import HifDailyProducer
                    self.daily = HifDailyProducer(self)
                return self.daily.observe(reconciliation=reconciliation)
            if self.supply is None:
                from .supply import HifSupplyProducer
                self.supply = HifSupplyProducer(self)
            return self.supply.observe()
        controller = self.context.tasker.controller
        pending = not reconciliation and self.host is not None and self.host.hooks.transition_gate.pending is not None
        deadline = self.clock.now() + (self.host.hooks.transition_gate.timeout_ms if pending else 8000) if _deadline is None else _deadline
        entry = self._round_one_entry if not pending and not reconciliation else None
        if entry is not None:
            deadline = entry[1]
        gate = self.host.hooks.transition_gate if self.host is not None else None
        cold_result = (reconciliation and gate is not None and gate.pending is not None
            and gate.pending.target.kind == 'round_one_result_next')
        if not reconciliation and self.host is not None and self.host.hooks.transition_gate.pending is not None:
            gate = self.host.hooks.transition_gate
            deadline = min(deadline, gate.issued_at_ms + gate.timeout_ms)
        from .first_day import deck_view_changed, initial_deck_scroll_needs_read
        settle_deadline = _deck_settle_deadline
        if _deadline is None and gate is not None and gate.pending is not None:
            from .prestart_search import MEMORY_SETTLE_SECONDS
            memory_delay = MEMORY_SETTLE_SECONDS.get(gate.pending.target.kind)
            if memory_delay:
                time.sleep(memory_delay)
        if settle_deadline is None and initial_deck_scroll_needs_read(self.host, gate):
            # hiro deck_review._still: the swipe completes before its initial
            # 0.8 s settle. Then poll the viewport at 0.3 s for at most 2.5 s.
            time.sleep(.8)
            settle_deadline = min(deadline, self.clock.now() + 2500)
        previous_view = None
        deck_moving = False
        while True:
            started = self.clock.now()
            if started >= deadline or self.context.tasker.stopping:
                raise ValueError('HIF current recovery capture deadline reached')
            capture_with_guard = getattr(controller, 'capture_with_guard', None)
            capture_with_record = getattr(controller, 'capture_with_record_guard', None)
            record_guard = self.capture_record_guard is not None and callable(capture_with_record)
            delegated_guard = record_guard or self.capture_guard is not None and callable(capture_with_guard)
            if self.capture_guard is not None and not delegated_guard and self.capture_guard() is not True:
                raise ValueError('HIF route capture budget guard was revoked')
            job = (capture_with_record(self.capture_record_guard) if record_guard
                else capture_with_guard(self.capture_guard) if delegated_guard else controller.post_screencap()).wait()
            if not job.succeeded:
                raise ValueError('HIF route native capture failed')
            image = job.get()
            key = f'route-frame-{self.game_pid}-{started}'
            self.current_frame_id = key
            try:
                page = self._page(image)
            except (ValueError, RuntimeError) as error:
                from .nonbattle_date import NonbattleDateObserved
                from .produce_settings import ProduceSettingsObserved
                if isinstance(error, (NonbattleDateObserved, ProduceSettingsObserved)): raise
                rejected = dict(frame_id=key, phase='unknown', reason=str(error), rows=self.rows)
                if isinstance(getattr(error, 'evidence', None), dict):
                    rejected['field_evidence'] = error.evidence
                    if isinstance(error.evidence.get('rows'), list):
                        rejected['rows'] = error.evidence['rows']
                self.events.append(rejected)
                if self.evidence_sink is not None:
                    self.evidence_sink(image, rejected)
                entry_transition = (entry is not None or cold_result) and isinstance(error, UnknownRecoveryPage)
                remaining = deadline - self.clock.now()
                if entry_transition and remaining <= 0:
                    raise ValueError('HIF current recovery capture deadline reached') from error
                if (entry_transition or pending and not gate.stopped) and remaining > 0:
                    time.sleep(min(.5 if entry is not None and started >= entry[0] else .1, remaining / 1000))
                    continue
                raise
            if isinstance(page.get('recovery_evidence'), dict):
                page['recovery_evidence']['frame_id'] = key
            if self.evidence_sink is not None:
                self.evidence_sink(image, dict(frame_id=key, phase=page['phase'], observation=page))
            if 'memory_stats' in page:
                if page['memory_stats'] is None:
                    if self.clock.now() + 1000 >= deadline:
                        raise ValueError('cannot read the selected qualifier memory on the start confirmation')
                    time.sleep(1)
                    continue
                if 'memory_expected' not in page:
                    page.update(memory_expected=page['memory_stats'], memory_matches=True)
            if settle_deadline is not None and page['phase'] == 'inventory':
                # After a swipe, watch the viewport for motion. Only the
                # resulting image is used for recognition or an action receipt.
                deck_moving = previous_view is not None and deck_view_changed(previous_view, image)
                if (previous_view is None or deck_moving) and self.clock.now() < settle_deadline:
                    previous_view = image
                    time.sleep(.3)
                    continue
            break
        if (entry is not None or cold_result) and self.clock.now() >= deadline:
            raise ValueError('HIF current recovery capture deadline reached')
        from .prestart import PRESTART_SCENES, MEMORY_LIST_SCENES, retain_preparation_page
        from .prestart_refill import AP_REFILL_CANCEL, AP_REFILL_CANCELLED, AP_REFILL_INSUFFICIENT, AP_REFILL_SCENES, ap_refill_target
        if (gate is not None and gate.pending is not None and gate.pending.target.kind == AP_REFILL_CANCEL
                and page['phase'] in ('start_home', 'start_hif_scenario')):
            # Closing the disabled AP prompt ends this task on the observed return page.
            page = dict(phase=AP_REFILL_CANCELLED, target=None)
        phase = page['phase']
        keys = (key,)
        if phase in PRESTART_SCENES:
            retain_preparation_page(self.host, page, tuple(keys))
            from .prestart_search import remember_search_page
            remember_search_page(self.host, page)
            self.preparation_page = page
        submit_start = False
        if phase == 'start_confirmation':
            from .start_transaction import available_start
            submit_start = available_start(self.host, page)
            if submit_start:
                page = dict(page, target=tuple(page['start_evidence']['button']))
        from .first_day import uses_initial_deck_observation
        initial_deck = phase == 'inventory' and uses_initial_deck_observation(self.host)
        if initial_deck:
            from .first_day import InitialDeckScrolling, observe_initial_deck
            try:
                inventory = observe_initial_deck(self, [image], [page['rows']], keys,
                    settle_deadline=settle_deadline, still_moving=deck_moving)
            except InitialDeckScrolling:
                return self.observe(reconciliation=reconciliation, _deadline=deadline, _deck_settle_deadline=settle_deadline)
            self.inventory_navigation = None
            if not inventory['read_finished']:
                self.inventory_navigation = dict(kind='recovery_inventory_scroll',
                    key=f'inventory-scroll:{inventory["displacement"]}', target=(350, 770, 1, 80))
                page['target'] = self.inventory_navigation['target']
        elif phase == 'inventory':
            pending = self.host.hooks.transition_gate.pending
            inspect_index = (int(pending.target.key.split(':')[1]) if pending is not None
                and pending.target.kind == 'recovery_inventory_detail' else None)
            try:
                inventory = self.inventory.observe(self.context, [image], [page['rows']],
                    keys, inspect_index=inspect_index,
                    reread_top=pending is None or pending.target.kind == 'recovery_inventory')
            except ValueError as error:
                from .new_run import start_record
                initial_readonly = (pending is None and self.host.ledger is None
                    and self.host.ownership['phase'] == 'recovery'
                    and (start_record(self.host.handoff) or {}).get('status') == 'complete')
                if ((initial_readonly or pending is not None and pending.target.kind in ('recovery_inventory', 'recovery_inventory_scroll'))
                        and (reconciliation or not self.host.hooks.transition_gate.stopped)
                        and str(error) in ('inventory scroll has no unique overlapping translation',
                            'flat inventory scroll has not settled', 'inventory customization badge is ambiguous')
                        and self.clock.now() < deadline):
                    self.events.append(dict(phase='unstable_inventory_motion', frame_ids=list(keys), reason=str(error)))
                    return self.observe(reconciliation=reconciliation, _deadline=deadline)
                raise
            self.inventory_navigation = self.inventory.next_navigation()
            self.host.handoff = dict(self.host.handoff or {}, nonbattle_inventory=inventory)
            if self.inventory_navigation is not None:
                page['target'] = self.inventory_navigation['target']
        if phase == 'identity':
            idol, plan, _, maximum = page['identity']
            self.identity = dict(run_id=self.run_id, revision=self.clock.now(),
                idol=idol, plan=plan, max_stamina=maximum)
            self.identity_frames = tuple(keys)
            self.host.handoff = dict(self.host.handoff or {}, recovery_identity=self.identity,
                identity_frame_ids=list(keys), game_pid=self.game_pid)
            from .recovery_inventory import scope
            self.host.handoff['recovery_identity_scope'] = scope(self.host)
        if all(name in page for name in ('stamina', 'max_stamina', 'currency')):
            self.last_interval = (page['stamina'], page['max_stamina'], page['currency'])
        if phase in ('interval', 'start_created') and page.get('drink_slots') is not None:
            if phase == 'start_created':
                from .first_day import first_day_scope
                origin = first_day_scope(self.host)
                self.identity = dict(run_id=self.run_id, revision=origin['request']['source_revision'],
                    idol=origin['request']['identity'][0], plan=origin['request']['identity'][1],
                    max_stamina=page['max_stamina'])
                self.identity_frames = tuple(origin['request']['source_frame_ids'])
                self.host.handoff = dict(self.host.handoff or {}, recovery_identity=self.identity,
                    identity_frame_ids=list(self.identity_frames), game_pid=self.game_pid)
            from .ledger import HifLedger, HifLedgerState
            from .recovery_inventory import permanent_cards
            if phase == 'start_created' and page.get('inventory_opened') is False:
                from .ledger import PartialDeckLedgerState
                state_type, cards, card_count = PartialDeckLedgerState, (), None
                origin.setdefault('initial_deck', dict(opened=False, reason='deck_button_not_found', spotlight_plus=None,
                    total=None, identified=0, frame_ids=list(keys)))
                self.host.handoff['cultivation_start'] = origin
            else:
                inventory = ((self.host.handoff or {}).get('nonbattle_inventory')
                    if phase == 'start_created' and uses_initial_deck_observation(self.host) else self.current_inventory())
                state_type, card_count = HifLedgerState, inventory['total']
                if inventory.get('first_day_once') and not inventory['complete'] and phase == 'start_created':
                    from .ledger import PartialDeckLedgerState
                    from .recovery_inventory import partial_permanent_cards
                    cards = partial_permanent_cards(self.host, inventory)
                    state_type = PartialDeckLedgerState
                else:
                    cards = permanent_cards(self.host, inventory)
            if page.get('drink_inspect_slot') is None:
                from .recovery_drinks import resolve
                drinks = resolve(self.host, dict(kind='root', slots=page['drink_slots'],
                    **{key: page[key] for key in ('stamina', 'max_stamina', 'currency')}), self.terminal_remainder) if phase == 'interval' else ()
                state = state_type(self.run_id, self.clock.now(), 1 if phase == 'start_created' else 7,
                    'schedule' if phase == 'start_created' else 'interval', cards, card_count, drinks, len(drinks),
                    page['currency'], page['stamina'], page['max_stamina'], tuple(sorted(page['counters'].items())))
                self.complete_ledger = HifLedger(state, card_catalog=self.host._catalog, drink_ids=self.host._drink_ids,
                    settled=self.host.ledger.settled if self.host.ledger is not None else ())
                self.host.handoff = dict(self.host.handoff or {}, current_drink_slots=page['drink_slots'],
                    current_resource_frame_ids=list(keys))
        kind = {'interval': ('recovery_inventory' if page.get('inventory_navigation') else 'recovery_menu'),
            'menu': 'recovery_notebook', 'notebook': 'recovery_identity',
            'identity': 'recovery_close', 'inventory': 'recovery_close',
            'owned_items': 'recovery_close', 'round_one_result': 'round_one_result_next',
            'round_one_award': 'round_one_award_next', 'end_result': 'end_result_next',
            'ending_exit_confirmation': 'end_defeat_exit',
            'ending_summary': 'end_star_reward_next',
            'ending_generation': 'end_generate_memory',
            'ending_generated_memory': 'end_memory_complete_next',
            'ending_complete': 'end_memory_detail_next',
            'start_ap_menu': 'start_ap_save_home',
            'start_resume_home': 'start_open_current_save',
            'start_resume_confirm': 'start_resume_current_save',
            'start_home': 'start_open_produce', 'start_hif_scenario': 'start_select_hif',
            'start_idol': 'start_selected_idol_next'}.get(phase)
        if phase in ('start_idol', *MEMORY_LIST_SCENES):
            kind = page.get('target_kind', kind)
        if phase in AP_REFILL_SCENES:
            kind = ap_refill_target(page)
        from .prestart_search import search_target_key
        key = search_target_key(kind, page)
        if phase == 'interval' and page['target'] is None and self.complete_ledger is not None:
            key = None
        if phase == 'recovery_drink_detail':
            kind, key = 'recovery_drink_close', f'drink-close:{page["slot"]}'
        elif phase == 'interval' and page.get('drink_inspect_slot') is not None:
            kind, key = 'recovery_drink_detail', f'drink-detail:{page["drink_inspect_slot"]}'
        if submit_start:
            kind = key = 'start_hif_once'
        elif phase == 'start_created' and page.get('inventory_navigation'):
            kind = key = 'recovery_inventory'
        elif phase == 'start_created' and page['target'] is not None:
            kind = key = 'start_ap_open_menu'
        if phase == 'ending_summary':
            kind = key = {'star_reward': 'end_star_reward_next',
                'final_evaluation': 'end_final_evaluation_next',
                'produce_complete': 'end_produce_complete_next',
                'received_event_rewards': 'end_produce_complete_next',
                'event_points_result': 'end_produce_complete_next'}[page['summary_kind']]
        if phase == 'inventory' and self.inventory_navigation is not None:
            kind, key = (self.inventory_navigation['kind'], self.inventory_navigation['key'])
        if (kind is None and phase not in ('ending_home', 'start_unbound_selection', 'start_confirmation',
                'start_memory_search_stopped', 'start_created', 'start_ap_home', AP_REFILL_INSUFFICIENT, AP_REFILL_CANCELLED)
                and not (self.resume_probe_home and phase == 'resume_home')):
            raise ValueError('unexpected HIF recovery destination')
        revision = self.clock.now()
        fields = ':'.join(str(page.get(name, '')) for name in ('stamina', 'max_stamina', 'currency', 'identity'))
        if initial_deck:
            fields += f':initial-deck:{inventory["displacement"]}:{inventory["read_finished"]}'
        elif phase == 'inventory':
            # The next inspection may change as an unresolved card settles.
            # Like activity/reward census, identify the physical current view,
            # not its current plan or a one-pixel scroll registration.
            fields += f':{self.inventory.total}:{tuple(sorted(self.inventory.boxes))}:{self.inventory.selected_index}'
        elif phase == 'end_result':
            fields += f':{page["score"]}:{page["cumulative_score"]}:{page["outcome"]}'
        elif phase == 'recovery_drink_detail':
            fields += f':{page["slot"]}:{page["drink_id"]}'
        elif phase == 'ending_home':
            fields += f':ap:{page["ap"]}/{page["max_ap"]}'
        elif phase == 'ending_summary':
            if page['summary_kind'] == 'produce_complete':
                fields += ':produce_complete'
                if page.get('completion_stage'):
                    fields += ':' + page['completion_stage']
            elif page['summary_kind'] in ('received_event_rewards', 'event_points_result'):
                fields += ':' + page['summary_kind']
            else:
                fields += (f':star_reward:{page["star_total"]}:{page["star_increment"]}' if page['summary_kind'] == 'star_reward'
                    else ':final_evaluation')
        elif phase == 'round_one_award':
            fields += f':{page["award_name"]}'
        elif phase == 'ending_generation':
            fields += f':{page["generation_idol"]}'
        elif phase in PRESTART_SCENES:
            fields += f':ap:{page["ap"]}' if 'ap' in page else ''
            fields += f':{page.get("remaining_ap", "")}:{page.get("booster_selected", "")}'
            if 'memory_cell' in page:
                fields += f':memory:{page["memory_page"]}:{page["memory_cell"]}'
        self.last_notebook_day = page.get('day') if phase in ('notebook','identity') else None
        observation = Observation(self.run_id, revision, phase,
            f'recovery:{phase}:{fields}', None, 'unverified', 'unverified', revision - started, len(keys), 'HIF',
            (Target(key, kind, available=True),) if key is not None else ())
        self.events.append(dict(phase=phase, frame_ids=list(keys), capture_age_ms=revision - started))
        self.last_frame_ids = tuple(keys)
        if entry is not None:
            self._round_one_entry = None
        drink_proof = None
        if phase == 'interval' and page.get('drink_slots') is not None:
            drink_proof = dict(kind='root', slots=page['drink_slots'], frame_ids=list(keys),
                inspect_slot=page.get('drink_inspect_slot'), **{key: page[key] for key in ('stamina', 'max_stamina', 'currency')})
        elif phase == 'recovery_drink_detail':
            drink_proof = dict(kind='detail', slot=page['slot'], drink_id=page['drink_id'], frame_ids=list(keys),
                detail=deepcopy(page['drink_detail_evidence']))
        result = HifRoutePage(observation, {key: tuple(page['target'])} if key is not None else {}, frame_ids=tuple(keys),
            baseline=self.complete_ledger.state if phase in ('interval', 'start_created') and self.complete_ledger else None,
            closing=dict(page) if self.ending or phase == 'round_one_result' else None,
            start=dict(page) if phase in ('start_confirmation', 'start_created', 'start_ap_home',
                *MEMORY_LIST_SCENES, *AP_REFILL_SCENES) else None,
            drink_inventory=drink_proof)
        if phase == 'inventory' and (not initial_deck or not inventory['read_finished']):
            from .recovery_inventory import current_record
            from .nonbattle_inventory import retain_inventory_registration
            stored = current_record(self.host, retain_inventory_registration(self.host, result, inventory))
            self.host.handoff = dict(self.host.handoff or {}, nonbattle_inventory=stored)
        return result

    def prepare(self, gate):
        self.prepare_recovery_epoch()
        self.prepare_inventory_census()
        self.prepare_drink_remainder()
        if gate.pending is None:
            from .new_run import start_record
            start = start_record(self.host.handoff) or {}
            if self.ap_readback_report is not None and start.get('status') == 'created_ap_unverified':
                self.host.accept_start_ap(None, None, observed_report=self.ap_readback_report)
            return gate
        from .input_receipt import original_input_report
        try:
            automatic = original_input_report(self.host, gate)
        except ValueError:
            if self.recovery_report is None:
                raise
        else:
            self.recovery_report = automatic
        from .customization_transaction import request as customization_request
        if customization_request(self.host) is not None:
            if getattr(self, 'customization_producer', None) is None:
                from .customization_production import HifCustomizationProducer
                self.customization_producer = HifCustomizationProducer(self)
            return self.customization_producer.prepare(gate, self.recovery_report)
        from .activity_reward import request as activity_request
        if activity_request(self.host) is not None:
            if getattr(self, 'activity_producer', None) is None:
                from .activity_production import HifActivityProducer
                self.activity_producer = HifActivityProducer(self)
            return self.activity_producer.prepare(gate, self.recovery_report)
        from .rewards import active
        if active(self.host):
            if getattr(self, 'reward_producer', None) is None:
                from .reward_production import HifRewardProducer
                self.reward_producer = HifRewardProducer(self)
            return self.reward_producer.prepare(gate, self.recovery_report)
        if gate.pending.target.kind == 'round_one_result_next':
            proof = self.observe(reconciliation=True)
            return self.host.reconcile_round_one_return(gate, self.recovery_report, proof, self.last_interval)
        if gate.pending.target.kind == 'recovery_inventory':
            proof = self.observe(reconciliation=True)
            from .first_day import uses_initial_deck_observation
            if gate.pending.source.scene == 'start_created' and uses_initial_deck_observation(self.host):
                from .prestart import reconcile_preparation_navigation
                return reconcile_preparation_navigation(self.host, gate, self.recovery_report, proof)
            return self.host.reconcile_inventory_entry(gate, self.recovery_report, proof, self.host.handoff['nonbattle_inventory'])
        if gate.pending.target.kind in ('recovery_drink_detail', 'recovery_drink_close'):
            proof = self.observe(reconciliation=True)
            return self.host.reconcile_recovery_drink(gate, self.recovery_report, proof)
        if self.host.ownership['phase'] == 'nonbattle' and self.host.ledger.state.day in (3, 6):
            if getattr(self, 'activity_producer', None) is None:
                from .activity_production import HifActivityProducer
                self.activity_producer = HifActivityProducer(self)
            return self.activity_producer.prepare(gate, self.recovery_report)
        from .prestart import PRESTART_NAVIGATION, reconcile_preparation_navigation
        from .first_day import FIRST_DAY_CONTRACTS, uses_initial_deck_observation
        from .start_transaction import START_AP_KIND, START_RESUME_CONTRACTS, START_RECEIPT_CONTRACTS, reconcile_started
        if gate.pending.target.kind == START_AP_KIND:
            proof = self.observe(reconciliation=True)
            return reconcile_started(self.host, gate, self.recovery_report, proof)
        if (self.host.ledger is None and gate.pending.target.kind in FIRST_DAY_CONTRACTS
                and (gate.pending.target.kind != 'recovery_inventory_scroll' or uses_initial_deck_observation(self.host))
                and gate.pending.source.scene == FIRST_DAY_CONTRACTS[gate.pending.target.kind][0]):
            proof = self.observe(reconciliation=True)
            return reconcile_preparation_navigation(self.host, gate, self.recovery_report, proof)
        if gate.pending.target.kind in (*PRESTART_NAVIGATION, *START_RECEIPT_CONTRACTS, *START_RESUME_CONTRACTS):
            proof = self.observe(reconciliation=True)
            return reconcile_preparation_navigation(self.host, gate, self.recovery_report, proof)
        if self.host.ownership['phase'] == 'nonbattle':
            if self.host.ledger.state.day < 7:
                if self.daily is None:
                    from .daily import HifDailyProducer
                    self.daily = HifDailyProducer(self)
                return self.daily.prepare(gate, self.recovery_report)
            if self.supply is None:
                from .supply import HifSupplyProducer
                self.supply = HifSupplyProducer(self)
            return self.supply.prepare(gate, self.recovery_report)
        if gate.pending.target.kind in ('round_one_award_next', 'end_result_next', 'end_defeat_exit', 'end_star_reward_next', 'end_final_evaluation_next',
                'end_generate_memory', 'end_memory_complete_next', 'end_memory_detail_next', 'end_produce_complete_next'):
            from .engine import completion_next_page
            if gate.pending.target.kind == 'end_produce_complete_next' and completion_next_page(gate.pending.source):
                time.sleep(max(0, 1.2 - (self.clock.now() - gate.issued_at_ms) / 1000))
            proof = self.observe(reconciliation=True)
            return self.host.reconcile_ending_navigation(gate, self.recovery_report, proof,
                terminal_report=self.terminal_report)
        if gate.pending.target.kind == 'recovery_inventory_detail':
            retained_handoff = self.host.handoff
            proof = self.observe(reconciliation=True)
            self.host.handoff = retained_handoff
            return self.host.reconcile_inventory_detail(gate, self.recovery_report, proof, self.inventory.record())
        if gate.pending.target.kind == 'recovery_inventory_scroll':
            retained_handoff = self.host.handoff
            proof = self.observe(reconciliation=True)
            self.host.handoff = retained_handoff
            return self.host.reconcile_inventory_scroll(gate, self.recovery_report, proof, self.inventory.record())
        if not gate.stopped:
            return gate
        # New read-only postmortem evidence may be collected after the old
        # failure. The original issued time/deadline stays in its failure receipt.
        proof = self.observe(reconciliation=True)
        result = self.host.reconcile_recovery_navigation(gate, self.recovery_report, proof, self.last_interval)
        self.events.append(dict(phase='recovery_navigation_reconciled', source_result=self.recovery_report['id'],
            original_issued_at_ms=gate.issued_at_ms, receipt_frame_ids=list(proof.frame_ids)))
        return result

    def decide(self, observation, config):
        error = validate_observation(observation)
        if error:
            return Decision(Outcome.STOP, error)
        if observation.scene == 'round_one_result':
            from .ending import ROUND_ONE_RESULT_DESTINATIONS
            return propose_unique(observation, observation.targets, 'hif-accepted-round-one-result-return',
                ROUND_ONE_RESULT_DESTINATIONS)
        if observation.scene == 'recovery_drink_detail':
            return propose_unique(observation, observation.targets, 'hif-current-recovery-drink-cancel-only', ('interval',))
        if observation.scene == 'start_created':
            from .start_transaction import _budget
            if self.host is not None and _budget(self.host)['status'] == 'complete':
                from .first_day import first_day_scope
                start = first_day_scope(self.host)
                if self.complete_ledger is not None:
                    from .ledger import PartialDeckLedgerState
                    partial = isinstance(self.complete_ledger.state, PartialDeckLedgerState)
                    self.host.bind_recovered_ledger(self.complete_ledger, frame_ids=self.last_frame_ids)
                    return Decision(Outcome.PAUSE, 'hif_current_initial_partial_ledger_observed' if partial
                        else 'hif_current_initial_complete_ledger_observed')
                if not observation.targets and observation.revision == start['creation_receipt']['revision']:
                    # The submission was just settled on a creation-only observation.
                    # The next automatic observation reads the normal inventory button.
                    return Decision(Outcome.PAUSE, 'hif_new_cultivation_created')
                return propose_unique(observation, tuple(target for target in observation.targets
                    if target.kind == 'recovery_inventory'), 'hif-first-day-current-inventory', ('inventory',))
            if (self.host is not None and _budget(self.host)['status'] == 'created_ap_unverified'
                    and any(target.kind == 'start_ap_open_menu' for target in observation.targets)):
                return propose_unique(observation, tuple(target for target in observation.targets
                    if target.kind == 'start_ap_open_menu'), 'hif-open-menu-for-actual-ap-receipt', ('start_ap_menu',))
            return Decision(Outcome.PAUSE, 'hif_new_cultivation_created_actual_ap_receipt_required')
        if observation.scene == 'start_ap_menu':
            from .start_transaction import _budget
            if self.host is None or _budget(self.host)['status'] != 'created_ap_unverified':
                return Decision(Outcome.STOP, 'hif_ap_receipt_menu_requires_current_created_run')
            return propose_unique(observation, tuple(target for target in observation.targets
                if target.kind == 'start_ap_save_home'), 'hif-save-current-run-for-ap-readback', ('start_ap_home',))
        if observation.scene == 'start_ap_home':
            from .start_transaction import _budget
            within_quote = self.host._page.start['ap'] >= _budget(self.host)['request']['quoted_ap_after']
            if within_quote or self.ap_clock_reports is not None or self.ap_readback_report is not None:
                self.host.accept_start_ap(self.host._page, self.ap_clock_reports,
                    observed_report=self.ap_readback_report)
                return Decision(Outcome.PAUSE, 'hif_actual_start_ap_verified_resume_current_save')
            return Decision(Outcome.PAUSE, 'hif_current_post_start_ap_observed_requires_receipt')
        if observation.scene in ('start_resume_home', 'start_resume_confirm'):
            from .start_transaction import START_RESUME_CONTRACTS, _budget
            if self.host is None or _budget(self.host)['status'] != 'complete':
                return Decision(Outcome.STOP, 'hif_current_save_requires_original_ap_and_creation_receipt')
            kind = 'start_open_current_save' if observation.scene == 'start_resume_home' else 'start_resume_current_save'
            return propose_unique(observation, tuple(target for target in observation.targets if target.kind == kind),
                'hif-resume-only-current-first-day', START_RESUME_CONTRACTS[kind][1])
        from .prestart import PRESTART_SCENES, decide_preparation
        if observation.scene in PRESTART_SCENES:
            if self.preparation_page is None or self.preparation_page['phase'] != observation.scene:
                return Decision(Outcome.STOP, 'hif_start_current_page_not_bound')
            return decide_preparation(self.host, observation, self.preparation_page)
        if self.ending and observation.scene == 'end_result':
            from .ending import ROUND_TWO_RESULT_DESTINATIONS
            return propose_unique(observation, observation.targets, 'hif-accepted-round-two-result-next',
                ROUND_TWO_RESULT_DESTINATIONS)
        if self.ending and observation.scene == 'round_one_award':
            from .ending import STAR_REWARD_DESTINATIONS
            return propose_unique(observation, observation.targets, 'hif-obtained-round-one-award-acknowledgement',
                ('interval', 'round_one_award', *STAR_REWARD_DESTINATIONS))
        if self.ending and observation.scene == 'ending_exit_confirmation':
            from .ending import DEFEAT_EXIT_DESTINATIONS
            return propose_unique(observation, observation.targets, 'hif-defeat-end-without-rechallenge',
                DEFEAT_EXIT_DESTINATIONS)
        if self.ending and observation.scene == 'ending_summary':
            from .ending import FINAL_EVALUATION_DESTINATIONS, PRODUCE_COMPLETION_DESTINATIONS, star_reward_destinations
            if any(target.kind == 'end_produce_complete_next' for target in observation.targets):
                from .closing_state import completed_memory
                if self.host is None or completed_memory(self.host, self.terminal_report) is None:
                    return Decision(Outcome.PAUSE, 'hif_produce_completion_requires_original_memory_receipt')
                return propose_unique(observation, tuple(target for target in observation.targets
                    if target.kind == 'end_produce_complete_next'), 'hif-produce-completion-next',
                    PRODUCE_COMPLETION_DESTINATIONS)
            targets = tuple(target for target in observation.targets
                if target.kind in ('end_star_reward_next', 'end_final_evaluation_next'))
            expected = (FINAL_EVALUATION_DESTINATIONS if len(targets) == 1
                and targets[0].kind == 'end_final_evaluation_next'
                else star_reward_destinations((self.host.handoff or {}).get('round_number')))
            return propose_unique(observation, targets, 'hif-summary-acknowledgement', expected)
        if self.ending and observation.scene == 'ending_generation':
            if self.host is not None:
                from .closing_state import completed_memory
                if completed_memory(self.host, self.terminal_report) is not None:
                    return Decision(Outcome.PAUSE, 'hif_memory_already_completed_no_regeneration')
            from .ending import MEMORY_GENERATION_DESTINATIONS
            return propose_unique(observation, tuple(target for target in observation.targets
                if target.kind == 'end_generate_memory'), 'hif-submit-current-memory-generation',
                MEMORY_GENERATION_DESTINATIONS)
        if self.ending and observation.scene == 'ending_generated_memory':
            from .ending import MEMORY_COMPLETION_DESTINATIONS
            from .closing_state import completed_memory
            if self.host is None or completed_memory(self.host, self.terminal_report) is None:
                return Decision(Outcome.PAUSE, 'hif_generated_memory_requires_original_submission_receipt')
            return propose_unique(observation, tuple(target for target in observation.targets
                if target.kind == 'end_memory_complete_next'), 'hif-acknowledge-generated-current-memory',
                MEMORY_COMPLETION_DESTINATIONS)
        if self.ending and observation.scene == 'ending_complete':
            from .ending import MEMORY_DETAIL_DESTINATIONS
            from .closing_state import completed_memory
            if self.host is None or completed_memory(self.host, self.terminal_report) is None:
                return Decision(Outcome.PAUSE, 'hif_memory_details_require_original_completion_receipt')
            return propose_unique(observation, tuple(target for target in observation.targets
                if target.kind == 'end_memory_detail_next'), 'hif-current-generated-memory-details-next',
                MEMORY_DETAIL_DESTINATIONS)
        if self.ending and observation.scene == 'ending_home':
            if self.host is not None:
                from .closing_state import completed_memory
                if completed_memory(self.host, self.terminal_report) is not None:
                    self.host.accept_ending_home(self.host._page, terminal_report=self.terminal_report)
                    return Decision(Outcome.PAUSE, 'hif_cultivation_completed_with_memory')
            return Decision(Outcome.PAUSE, 'hif_home_requires_memory_completion_receipt')
        if observation.scene == 'inventory':
            from .first_day import uses_initial_deck_observation
            if uses_initial_deck_observation(self.host):
                if any(target.kind == 'recovery_inventory_scroll' for target in observation.targets):
                    return propose_unique(observation, observation.targets, 'hif-initial-inventory-scroll', ('inventory',))
                return propose_unique(observation, tuple(target for target in observation.targets
                    if target.kind == 'recovery_close'), 'hif-initial-inventory-close', ('start_created',))
        if observation.scene == 'inventory' and self.inventory is not None:
            if self.inventory.complete:
                from .new_run import start_record
                if self.host is not None and self.host.ledger is None and start_record(self.host.handoff) is not None:
                    from .first_day import first_day_scope
                    first_day_scope(self.host)
                    return propose_unique(observation, observation.targets, 'hif-initial-inventory-close', ('start_created',))
                return propose_unique(observation, observation.targets, 'hif-current-inventory-close', ('interval',))
            return propose_unique(observation, observation.targets, 'hif-current-flat-inventory', ('inventory',))
        if observation.scene in ('inventory', 'owned_items'):
            return Decision(Outcome.PAUSE, 'hif_current_inventory_collection_required')
        if observation.scene == 'interval' and self.identity is not None:
            if self.complete_ledger is not None:
                self.host.bind_recovered_ledger(self.complete_ledger, frame_ids=self.last_frame_ids)
                return Decision(Outcome.PAUSE, 'hif_current_complete_nonbattle_ledger_observed')
            if any(target.kind == 'recovery_drink_detail' for target in observation.targets):
                return propose_unique(observation, observation.targets, 'hif-current-recovery-physical-drink-detail', ('recovery_drink_detail',))
            return propose_unique(observation, tuple(target for target in observation.targets
                if target.kind == 'recovery_inventory'), 'hif-current-inventory', ('inventory', 'owned_items'))
        expected = {'interval': ('menu',), 'menu': ('notebook', 'identity'),
            'notebook': ('identity',), 'identity': ('interval',)}
        if observation.scene not in expected:
            return Decision(Outcome.STOP, 'unknown_hif_recovery_scene')
        return propose_unique(observation, observation.targets,
            f'hif-current-recovery:{observation.scene}', expected[observation.scene])

    def decide_nonbattle(self, observation, config):
        if self.host.ledger.state.day in (3, 6) and getattr(self, 'activity_producer', None) is not None:
            return self.activity_producer.decide(observation, config)
        if self.host.ledger.state.day < 7 and self.daily is not None:
            return self.daily.decide(observation, config)
        if self.supply is None:
            page, state, gate = self.host._page, self.host.ledger.state, self.host.hooks.transition_gate
            # The sequence asks about its last receipt immediately after the
            # confirmed D6 exit changed the ledger to D7. That receipt has no
            # next input target and belongs to the previous producer. Continue
            # the same invocation with a fresh, normally assembled D7 page.
            if (state.day == 7 and state.phase == 'interval' and self.host.ledger.pending is None
                    and gate.pending is None and not gate.stopped and page is not None
                    and page.observation == observation and observation.scene == 'battle_intro'
                    and not observation.targets and page.baseline is not None
                    and page.baseline.day == 6 and page.baseline.phase == 'consult_shop'
                    and page.receipt == state and page.receipt_frame_ids == page.frame_ids
                    and source_frames_match(observation, page.frame_ids) and (page.activity or {}).get('kind') == 'battle_intro'):
                return Decision(Outcome.PAUSE, 'hif_verified_day_seven_route_boundary')
            return Decision(Outcome.STOP, 'current_interval_producer_not_bound')
        return self.supply.decide(observation, config)

    def customization_menu(self, page, config):
        if self.supply is None:
            from .supply import HifSupplyProducer
            self.supply = HifSupplyProducer(self)
        return self.supply.customization_menu(page, config)


def bind_hif_nonbattle_runtime(context, *, path, run_id, game_pid, client_size,
        authorise_capture, authorise_input, evidence_sink=None, recovery_report=None, terminal_report=None, ap_clock_reports=None,
        terminal_runtime=None, authorise_preparation=None):
    """Trusted client binding; callable permission gates are not task parameters."""
    import csv

    from utils.maa_input import verify_maa_input_endpoints
    from live_decision.catalog import SkillCardFeatureCatalog
    from live_decision.hif_capture_adapter import adapt_hif_context

    from .customization import CustomizationCatalog
    if not callable(authorise_capture) or not callable(authorise_input) or authorise_capture() is not True:
        raise ValueError('current HIF capture permission is not bound')
    if authorise_preparation is not None and not callable(authorise_preparation):
        raise ValueError('current HIF preparation permission is not callable')
    context = adapt_hif_context(context)
    if context.tasker.controller.set_screenshot_target_long_side(1280) is not True:
        raise ValueError('HIF route capture long-side option was not applied')
    size = tuple(client_size)
    context.gkh_hif_client_size = size

    def endpoint():
        verified = verify_maa_input_endpoints(context.tasker.controller.info, expected_client_size=size)
        if verified['game']['pid'] != game_pid:
            raise ValueError('current HIF input game process changed')
        return verified

    endpoint()
    clock = windows_route_clock()
    observer = HifRecoveryProducer(context, run_id=run_id, game_pid=game_pid,
        clock=clock, evidence_sink=evidence_sink, recovery_report=recovery_report, terminal_report=terminal_report,
        ap_clock_reports=ap_clock_reports, terminal_runtime=terminal_runtime)
    data = resolve_runtime_path(Path('assets/data'), Path('data'))
    catalog = SkillCardFeatureCatalog.load(data / 'skill_cards.csv')
    customization = CustomizationCatalog.load(data / 'skill_cards.csv', data / 'customizations.json')
    with (data / 'p_drinks.csv').open(encoding='utf-8-sig', newline='') as stream:
        drinks = {row['id'] for row in csv.DictReader(stream)}
    host = HifRouteHost(path, run_id=run_id, observe=observer.observe,
        authorise_capture=authorise_capture,
        # Product preparation checks the current task/run/owner. Its guarded
        # Context still verifies every actual input endpoint at SDK dispatch.
        # Other callers retain the original complete permission gate.
        authorise_input=(authorise_preparation if authorise_preparation is not None
            else lambda: authorise_input() is True and bool(endpoint())),
        clock=clock, new_verified_run=not Path(path).exists(),
        ownership=(dict(revision=1, owner='task600', phase='recovery', game_pid=game_pid)
            if not Path(path).exists() else None),
        input_owner='task600', game_pid=game_pid, recovery_decide=observer.decide,
        card_catalog=catalog, drink_ids=drinks,
        ending_terminal_report=terminal_report,
        customization_catalog=customization,
        customization_menu_source=observer.customization_menu,
        nonbattle_decide=observer.decide_nonbattle,
        recovery_prepare=observer.prepare)
    observer.host = host
    host._input_source = observer.freeze_input_source
    host._input_adapter = context.gkh_hif_capture_adapter
    context.gkh_hif_route_host = host
    context.gkh_hif_route_producer = observer
    return context
