"""Complete one unresolved opening thumbnail from the same inventory and route ledger."""
from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
from collections import Counter
from dataclasses import asdict

import cv2

from .inventory import (
    INVENTORY_ZONES,
    InventorySection,
    InventoryThumbnailRecognizer,
    frame_identifier,
    read_inventory_sections,
    _inventory_visible_boxes,
    corroborate_inventory_marker,
    _align_inventory_physical_rows,
)
from .opening_observer import valid_opening_frame_ids
from .inventory_customization import bind_inventory_customizations, inspect_thumbnail_customizations


def _opening_frames(source, frame_ids, label):
    paths = {item['frame_id']: item['path'] for item in source['frames'] if item['label'] == label}
    if not valid_opening_frame_ids(frame_ids):
        raise ValueError('archived opening observation needs its original frames')
    frames = [cv2.imread(paths[key]) if key in paths else None for key in frame_ids]
    if any(frame is None or frame_identifier(frame) != key for key,frame in zip(frame_ids, frames, strict=True)):
        raise ValueError('archived opening pixels are missing or changed')
    return frames


def reobserve_partial_inventory_return(source, module, context):
    from .hif_navigation_recovery import verify_unstarted_core

    panel = source['live_events'][0]['inventory_events'][0]
    params = source['formal_params']
    options = dict(identity_threshold=params['debug_identity_threshold'], identity_min_margin=params['debug_identity_min_margin'])
    recognizer = module.EmbeddingCardRecognizer.load(module.MODEL_ROOT)
    before_frames = _opening_frames(source, panel['before_frame_ids'], 'ProduceRecognitionCards')
    after_frames = _opening_frames(source, panel['return_observations'][-1]['frame_ids'], 'ProduceRecognitionCards')
    previous_art = getattr(context, 'gkh_hif_card_art', None)
    context.gkh_hif_card_art = module.FrozenMaskedArt.load(recognizer, module.MODEL_ROOT / 'hif')
    try:
        before = module._inventory_live_checkpoint(context, before_frames, recognizer,
            single_frame=len(before_frames) == 1, **options)
        after = module._inventory_live_checkpoint(context, after_frames, recognizer, inventory_before=before,
            single_frame=len(after_frames) == 1, **options)
    finally:
        context.gkh_hif_card_art = previous_art
    verify_unstarted_core(source['live_events'][-1]['initial_core'], before)
    same = verify_unstarted_core(before, after)
    return dict(same, source_result=source['id'], source='archived_inventory_return',
                fields=after['fields'], families=after['families'], input_click_count=0)


def reobserve_partial_inventory_detail(source, module, context):
    from .inventory_support import resolve_inventory_detail
    from .inventory_detail_badge import read_inventory_detail_badge_count
    from .inventory_customization import load_customization_catalog, read_inventory_detail_text_rows

    panel = source['live_events'][0]['inventory_events'][0]
    inputs = panel['inputs']
    rejections = panel.get('detail_rejections', ())
    selections = panel.get('customization_preselection', ())
    if not rejections or not selections or inputs[-2]['phase'] != 'customization_select':
        return None
    selected, rejected = selections[-1], rejections[-1]
    card_id = rejected['card_id']
    candidates = [row for row in panel['customization_presence']
                  if row['card_id'] == card_id and row['presence'] == 'DETAIL_CANDIDATE']
    if (len(candidates) != 1 or selected['observed_card_id'] != card_id or selected['card_id'] != card_id
            or selected['current_box'] != inputs[-2]['box'] or rejected['reason'] != panel['read_failure']):
        return None
    ids = rejected['frame_ids']
    frames = _opening_frames(source, ids, 'inventory-failure')
    catalog = load_customization_catalog(module.CATALOG_PATH, module._resolve_runtime_path(
        Path('assets/data/customizations.json'), Path('data/customizations.json')))
    observations = [read_inventory_detail_text_rows(context, frame) for frame in frames]
    try:
        detail = resolve_inventory_detail(catalog, candidates[0], observations, ids,
            badge_observer=lambda: read_inventory_detail_badge_count(context, frames, ids))
    except ValueError:
        return None
    detail['opening_detail_source'] = dict(source_result=source['id'], frame_ids=list(ids),
        run_id=source['observed_live_context']['run_id'], revision=source['observed_live_context']['revision'])
    return detail

def inventory_awaiting_return(source, ledger):
    """Rebind the finished card table. A fresh battle checkpoint completes its return."""
    inventory, opening = source['live_events']
    zone, gate = inventory['inventory_events']
    returns = zone.get('return_observations') or []
    reasons = ('inventory navigation Live visual family is unresolved', 'inventory navigation Live resources are unresolved')
    if (not returns or zone.get('reason') != returns[-1].get('reason')
            or any(r.get('accepted') is not False or r.get('reason') not in reasons for r in returns)
            or zone.get('read_failure') is not None or zone.get('after_frame_ids') or zone.get('live_state_unchanged') is not None
            or opening.get('failure_events') != inventory['inventory_events'] or gate.get('reason') != zone.get('reason')):
        raise ValueError('inventory return has no outstanding battle observation')
    composition = zone.get('composition') or {}
    counts = {name: len(ids) for name, ids in composition.items()}
    pages = zone.get('pages') or []
    total = sum(counts.values())
    if (set(counts) != set(INVENTORY_ZONES) or counts != zone.get('complete_zone_counts')
            or not pages or pages[-1].get('complete') is not True
            or total != pages[-1].get('observed_slot_count') or pages[-1].get('total') != total):
        raise ValueError('closed inventory lacks the complete observed card table')
    cards = bind_inventory_customizations(composition, zone.get('customization_presence', []),
                                          zone.get('detail_observations', []))
    identity = source['automatic_battle']['identity']
    if ledger is None:
        if identity['ledger_revision'] is not None:
            raise ValueError('closed inventory lacks its recorded route ledger')
    else:
        observed = Counter((c['card_id'], tuple(sorted(c['customizations'].items()))) for c in cards)
        expected = Counter((c['card_id'], tuple(sorted(map(tuple, c['customization'])))) for c in ledger['cards'])
        if ((identity['route_run_id'], identity['ledger_revision']) != (ledger['run_id'], ledger['revision'])
                or observed != expected):
            raise ValueError('closed inventory card definitions differ from the same-run ledger')
    run = source['observed_live_context']
    return dict(phase='inventory_permanent_card_state', cards=cards, zone_counts=counts,
        run_id=run['run_id'], revision=run['revision'], before_frame_ids=zone['before_frame_ids'],
        input_click_count=0, temporary_effects_bound=False, strategy_state_bound=False,
        recovered_inventory={'source': source['id']})


def corroborate_last_inventory_card(identities, boxes, batch, frames, reader, *, plan, total, ledger_ids):
    """Check the ledger's sole remaining card against the observed artwork and upgrade marker."""
    if not ledger_ids or len(ledger_ids) != total or len(batch) != len(boxes) or not frames:
        return batch, None
    missing = [index for index, row in enumerate(batch) if row['card_id'] is None]
    if len(missing) != 1:
        return batch, None
    index = missing[0]
    key = tuple(boxes)[index]
    known = dict(identities)
    for slot, row in zip(boxes, batch, strict=True):
        cid = row['card_id']
        if cid is not None:
            if known.setdefault(slot, cid) != cid:
                return batch, None
    if key in known or len(known) != total - 1:
        return batch, None
    remaining = Counter(ledger_ids)
    remaining.subtract(known.values())
    if any(count < 0 for count in remaining.values()) or sum(remaining.values()) != 1:
        return batch, None
    cid, = remaining.elements()
    original = batch[index]['observations']
    resolved = corroborate_inventory_marker(reader, frames, boxes[key], original, cid, plan=plan,
        reason='accepted_with_complete_ledger_and_current_marker')
    if not resolved:
        return batch, None
    frame_ids = tuple(frame_identifier(frame) for frame in frames)
    if not valid_opening_frame_ids(frame_ids):
        return batch, None
    updated = list(batch)
    updated[index] = dict(batch[index], card_id=cid, observations=resolved)
    return updated, dict(source='complete_ledger_and_current_marker', zone=key[0], index=key[1],
        card_id=cid, box=boxes[key], inventory_frame_ids=frame_ids, visual_group=resolved[0].visual_group,
        original_observations=tuple(map(asdict, original)), resolved_observations=tuple(map(asdict, resolved)))


def _observed_sections(source, frames, selected_title):
    # A completed collector pass logs the two OCR nodes in this order for each
    # frame. Replay their actual rows and the original image for header geometry.
    nodes = ('LiveInventorySections', 'LiveInventoryZoneHeaders')
    calls = [row for row in source['live_raw_fields'] if row['node'] in nodes][-len(nodes) * len(frames):]
    if not frames or [row['node'] for row in calls] != list(nodes) * len(frames):
        raise ValueError('closed inventory lacks its final page header reads')
    ordered = iter(calls)
    def recognition(node, image, **_params):
        row = next(ordered, None)
        if row is None or row['node'] != node:
            raise ValueError('closed inventory header needs a different OCR observation')
        return SimpleNamespace(hit=True, filtered_results=[SimpleNamespace(**r) for r in row['raw']['filtered']])
    recorded_context = SimpleNamespace(run_recognition=recognition)
    return [read_inventory_sections(recorded_context, frame, selected_card_title=selected_title) for frame in frames]


def recover_closed_inventory(source, module, context, confirmed_deck):
    """Return a derived cards component. The caller verifies original navigation and current Live continuity."""
    from arena_winrate.badge_reference import BadgeReferenceGallery

    identity = source['automatic_battle']['identity']
    if (source['handoff_id'] != identity['handoff_id'] or identity['route_run_id'] != confirmed_deck['run_id']
            or identity['ledger_revision'] != confirmed_deck['revision']):
        raise ValueError('closed inventory belongs to another handoff ledger')
    inventory, opening = source['live_events']
    if (inventory.get('event') != 'live_inventory_observe' or inventory.get('accepted') is not False
            or opening.get('event') != 'live_opening_state' or opening.get('accepted') is not False):
        raise ValueError('closed inventory is not a failed first opening observation')
    zone, gate = inventory['inventory_events']
    reason = 'inventory page contains an unresolved card'
    if (zone.get('phase') != 'inventory_zone_observation' or zone.get('read_failure') != reason
            or zone.get('live_state_unchanged') is not True or gate.get('reason') != reason
            or opening.get('failure_events') != inventory['inventory_events']):
        raise ValueError('closed inventory lacks an unchanged return from the unread page')
    ledger = confirmed_deck['cards']
    total = len(ledger)
    known, sections = {}, {}
    for page in zone['pages']:
        if page['total'] != total:
            raise ValueError('closed inventory total differs from its complete ledger')
        for section in page['sections']:
            name, count = section['name'], section['count']
            x, y, w, h = section['box']
            observed = InventorySection(name, count, (x, y + page['displacement'], w, h))
            if name not in INVENTORY_ZONES or sections.setdefault(name, observed).count != count:
                raise ValueError('closed inventory has conflicting section counts')
        for name, index, cid, _ in page['slots']:
            if not 0 <= index < sections[name].count or known.setdefault((name, index), cid) != cid:
                raise ValueError('closed inventory contains conflicting physical slots')
        if len(known) != page['observed_slot_count']:
            raise ValueError('closed inventory page coverage changed')

    rejected = zone['page_rejections'][-1]
    if rejected['reason'] != reason or not valid_opening_frame_ids(rejected['frame_ids']):
        raise ValueError('closed inventory final rejected pixels are incomplete')
    paths = {row['frame_id']: row['path'] for row in source['frames'] if row['label'] == 'inventory-failure'}
    frames = [cv2.imread(paths[key]) if key in paths else None for key in rejected['frame_ids']]
    if any(frame is None or frame_identifier(frame) != key for key, frame in zip(rejected['frame_ids'], frames)):
        raise ValueError('closed inventory final page pixels are missing or changed')
    live = source['observed_live_context']
    last_detail = zone['detail_observations'][-1] if zone['detail_observations'] else None
    selected_title = module.SkillCardFeatureCatalog.load(module.CATALOG_PATH).entries[last_detail['card_id']].name if last_detail else None
    counts = {name: section.count for name, section in sections.items()}
    direct = _observed_sections(source, frames, selected_title)
    for observed_total, headers in direct:
        if observed_total != total or [(h.name, h.count) for h in headers] != [(h.name, h.count) for h in direct[0][1]]:
            raise ValueError('closed inventory final section counts changed')
        for header in headers:
            if counts.setdefault(header.name, header.count) != header.count:
                raise ValueError('closed inventory final section contradicts its earlier page')
    if (sum(counts.values()) != total or not counts.get('山札') or counts.get('捨札') != 0
            or counts.get('除外', 0) or counts.get('手札') != len(opening['initial_core']['families'])):
        raise ValueError('closed opening inventory cannot account for each zone')
    final = rejected['detail']
    boxes = _align_inventory_physical_rows(frames, _inventory_visible_boxes(sections, final['displacement']))
    rows = final['slots']
    if [(r['zone'], r['index'], tuple(r['box'])) for r in rows] != [(z, i, b) for (z, i), b in boxes.items()]:
        raise ValueError('closed inventory final slots differ from the registered physical rows')
    reader = InventoryThumbnailRecognizer(BadgeReferenceGallery.load(module._resolve_runtime_path(
        Path('assets/resource/base/model/embedding/arena_badge_reference'),
        Path('resource/base/model/embedding/arena_badge_reference'))), module.EmbeddingCardRecognizer.load(module.MODEL_ROOT))
    batch = reader.identify_batch(frames, list(boxes.values()), plan=live['plan'])
    if any(before['card_id'] != fresh['card_id'] or any(old['reason'] != observation.reason
            or old['visual_group'] != observation.visual_group for old, observation in
            zip(before['observations'], fresh['observations'], strict=True)) for before, fresh in zip(rows, batch, strict=True)):
        raise ValueError('closed inventory pixels do not reproduce the original page observations')
    batch, proof = corroborate_last_inventory_card(known, boxes, batch, frames, reader, plan=live['plan'],
        total=total, ledger_ids=[row['card_id'] for row in ledger])
    if proof is None:
        raise ValueError('closed inventory last card is not supported by ledger, family and markers')
    final_slots = [(name, i, row['card_id'], boxes[name, i]) for (name, i), row in zip(boxes, batch, strict=True)]
    known.update({(name, i): cid for name, i, cid, _ in final_slots})
    composition = {name: [known[name, i] for i in range(counts.get(name, 0))] for name in INVENTORY_ZONES}
    badges = {(row['zone'], row['index']): row for row in zone['customization_presence']}
    for row in inspect_thumbnail_customizations(frames, final_slots):
        old = badges.setdefault((row['zone'], row['index']), row)
        if (old['card_id'], old['presence']) != (row['card_id'], row['presence']):
            raise ValueError('closed inventory final badge contradicts its previous observation')
    cards = bind_inventory_customizations(composition, list(badges.values()), zone['detail_observations'])
    actual = Counter((c['card_id'], tuple(sorted(c['customizations'].items()))) for c in cards)
    expected = Counter((c['card_id'], tuple(sorted(map(tuple, c['customization'])))) for c in ledger)
    if actual != expected:
        raise ValueError('closed inventory card definitions differ from the same-run ledger')
    return dict(phase='inventory_permanent_card_state', accepted=True, returned_unchanged=True,
        run_id=live['run_id'], revision=live['revision'], cards=cards,
        zone_counts={name: len(values) for name, values in composition.items()},
        before_frame_ids=zone['before_frame_ids'], after_frame_ids=zone['after_frame_ids'], input_click_count=0,
        temporary_effects_bound=False, strategy_state_bound=False, recovered_inventory=dict(
            source=source['id'], observed_counts=counts, zero_by_conservation=[z for z in INVENTORY_ZONES if z not in counts],
            final_card=proof))
