"""Resolve one visible reward row, its weak candidates, and the chosen physical slot."""
from __future__ import annotations

import re
from pathlib import Path
from functools import lru_cache
from itertools import product

import cv2
from utils.runtime_paths import resolve_runtime_path


@lru_cache(maxsize=1)
def load_offer_drink_reader(ids):
    from live_decision.drink_icons import EVENT_GREEN_JUICE_ID, load_default_drink_icons
    data = resolve_runtime_path(Path('assets/data/p_drinks.csv'), Path('data/p_drinks.csv'))
    folder = data.with_name('hif_drink_icons')
    if not folder.is_dir():
        folder = Path(__file__).resolve().parents[2] / '.local/runtime-data/hif_drink_icons'
    return load_default_drink_icons(folder, (*ids, EVENT_GREEN_JUICE_ID))


def _read_drink_offer_icon(reader, tile, ids):
    from live_decision.drink_icons import ICON_SIZE, MIN_SCORE, MIN_MARGIN

    # The 128px reward cell contains a centered 100px bottle image.
    height, width = tile.shape[:2]
    crop_size = 100
    left, top = width // 2 - crop_size // 2, height // 2 - crop_size // 2

    def crop_at(dx, dy):
        x, y = left + dx, top + dy
        patch = tile[y:y + crop_size, x:x + crop_size]
        return cv2.resize(patch, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)

    centered = reader.rank_crop(crop_at(0, 0))
    best, second = centered
    if best[1] >= MIN_SCORE and best[1] - second[1] >= MIN_MARGIN:
        return best[0] if best[0] in ids else None

    offsets = range(-4, 5, 2)
    shifted = [crop_at(dx, dy) for dx, dy in product(offsets, repeat=2)
               if (dx or dy) and 0 <= left + dx <= width - crop_size and 0 <= top + dy <= height - crop_size]
    if not shifted:
        return None
    scores = dict(centered)
    for pairs in reader._rank_crops(shifted):
        for drink_id, score in pairs:
            scores[drink_id] = max(score, scores.get(drink_id, -1.0))
    # Compare distinct drinks even when their best scores occur at different positions.
    best, second = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:2]
    return best[0] if best[0] in ids and best[1] >= MIN_SCORE and best[1] - second[1] >= MIN_MARGIN else None


def read_drink_offers(reader, image, ids, *, registered=None, focus=None):
    from .outing import outing_offer_boxes
    boxes = (tuple(tuple(row['box']) for row in registered) if registered is not None
        else outing_offer_boxes(image, drink=True))
    offers = []
    for index, (x, y, width, height) in enumerate(boxes):
        if focus is not None and index != focus:
            offers.append(dict(registered[index]))
            continue
        identity = None
        if reader is not None and reader._coverage_complete:
            identity = _read_drink_offer_icon(reader, image[y:y+height, x:x+width], ids)
        old = registered[index] if registered else {}
        if old.get('drink_id') and identity and old['drink_id'] != identity:
            raise ValueError('activity drink candidate changed at its original physical slot')
        row = dict(index=index, box=[x, y, width, height], drink_id=identity)
        if old.get('resolved_by_detail'):
            row['resolved_by_detail'] = True
        offers.append(row)
    return offers


def complete_offer(producer, page, image, rows):
    from . import activity_reward
    from .exchange import require_selection_corners
    from .production import _row
    from .activity_production import canonical, enabled_receive, read_card_offers

    host = producer.parent.host
    value = activity_reward.request(host)
    kind = page['kind']
    family = 'card' if kind.startswith('card_') else 'drink'
    id_field = family + '_id'
    inspecting, selected = kind.endswith('_inspection'), kind.endswith('_selected')
    registered = value.get('offers') if inspecting or selected else None
    focus = (value.get('inspecting_offer') if inspecting else value.get('selected_offer_index') if selected else None)
    if (selected or inspecting) and (type(focus) is not int or focus not in range(3) or not registered):
        raise ValueError('activity reward inspection lacks its original physical target')
    index = producer.select_offer_index(registered) if inspecting else focus
    if (inspecting and type(index) is int and index in range(len(registered)) and index != focus
            and registered[index].get(id_field) is not None):
        # A previously opened, unchosen slot can disclose its name. Its enlarged
        # artwork is not evidence against the already identified target elsewhere.
        offers = [dict(row) for row in registered]
    elif family == 'card':
        if producer.offer_reader is None:
            from .nonbattle_inventory import load_flat_inventory_readers
            producer.offer_reader = load_flat_inventory_readers()[0]
        offers = read_card_offers(producer.offer_reader, [image], host._catalog, registered=registered, focus=focus)
    else:
        reader = load_offer_drink_reader(tuple(sorted(producer.names.values())))
        offers = read_drink_offers(reader, image, producer.names.values(), registered=registered, focus=focus)
    if not inspecting and not selected:
        index = producer.select_offer_index(offers)
        if family == 'card':
            from .candidate_pool import maybe_edit_pool

            maybe_edit_pool(host, producer.parent.context, image, rows, offers, family='activity', default=index)
    if type(index) is not int or index not in range(len(offers)):
        raise ValueError('activity reward selection has no physical index')
    chosen = offers[index]
    page['offers'] = offers
    if inspecting:
        row = offers[focus]
        require_selection_corners(image, tuple(row['box']))
        if family == 'card':
            candidates = row.get('candidate_ids') or host._catalog.entries.keys()
            names = {identity: canonical(host._catalog.entries[identity].name) for identity in candidates}
        else:
            names = {identity: name for name, identity in producer.names.items()}
        try:
            title = _row(rows, '|'.join(re.escape(name) for name in names.values()), (100, 480, 520, 75))
            identities = [identity for identity, name in names.items() if name == title['text']]
            if len(identities) != 1 or row.get(id_field) not in (None, identities[0]):
                raise ValueError('activity reward detail contradicts its original candidate')
        except ValueError:
            if focus == index or chosen.get(id_field) is None:
                raise
            row[id_field] = None
            if family == 'card':
                row['upgraded'] = None
            page['inspected'] = dict(index=focus, **{id_field: None})
        else:
            detail = {id_field: identities[0]}
            if family == 'card':
                detail['upgraded'] = host._catalog.entries[identities[0]].upgraded
            row.update(detail, resolved_by_detail=True)
            page['inspected'] = dict(detail, index=focus)
    if selected and registered[focus].get(id_field) is not None and chosen.get(id_field) is None:
        # Enlargement may obscure the target thumbnail. Its visible name can confirm it.
        expected = registered[focus][id_field]
        name = (canonical(host._catalog.entries[expected].name) if family == 'card'
            else next(name for name, identity in producer.names.items() if identity == expected))
        _row(rows, re.escape(name), (100, 480, 520, 75))
        chosen.update(registered[focus])
    if chosen.get(id_field) is None:
        action = 'activity_inspect_' + family
        page.update(inspect_offer=index, target=tuple(chosen['box']),
            target_kind=action, target_key=action)
        return
    if family == 'card' and not value['cards'] and chosen['upgraded'] is not True:
        raise ValueError('activity first card reward contradicts its original upgraded-offer quote')
    page['selected_offer_index'] = index
    if not selected and not (inspecting and focus == index):
        action = 'activity_select_' + family
        page.update(target=tuple(chosen['box']), target_key=action, target_kind=action)
        return
    if selected and chosen[id_field] != registered[index].get(id_field):
        raise ValueError('selected activity offers changed from their original snapshot')
    require_selection_corners(image, tuple(chosen['box']))
    page['chosen'] = {id_field: chosen[id_field]}
    if family == 'card':
        page['chosen']['upgraded'] = chosen['upgraded']
    action = 'activity_receive_' + family
    page.update(target=enabled_receive(image, rows), target_kind=action, target_key=action)
