"""Read current lesson exchange offers with the existing thumbnail readers."""
import re
import math
import unicodedata
from collections import Counter
from dataclasses import asdict

import cv2


def exchange_offer_next(rows):
    """fb8 HIF CardSwapFlag reads Next in the bottom strip of the offer page."""
    from .production import _row

    try:
        return _row(rows, r'次[へヘ入]', (0, 1060, 720, 55), minimum=.3)
    except ValueError:
        return None


def exchange_dialogs_absent(image, rows):
    """Observe disappearance of both the replacement chooser and its confirmation."""
    from .prestart_memory import _reference

    if getattr(image, 'shape', ()) != (1280, 720, 3) or not cv2.meanStdDev(image)[1].any():
        return False
    for row in rows:
        text = row['text']
        x, y, _, _ = row['box']
        if re.search(r'チ[ェエヱ]ンジ(?:確認|するスキルカード)', text):
            return False
        if (y < 100 or x > 360 and y > 1080) and re.search(r'チ[ェエヱ]ンジ', text):
            return False
    for region, reference in ((image[:100, :300], 'exchange_icon'), (image[90:490], 'choose_move_cards')):
        scores = cv2.matchTemplate(region, _reference(reference), cv2.TM_CCOEFF_NORMED)
        if cv2.minMaxLoc(scores)[1] >= .8:
            return False
    return True


def exchange_offer_boxes(image):
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('lesson exchange capture layout is uncalibrated')
    # White offer panel isolates the whole opaque tile borders. Partial
    # entrance-animation cards cannot stand in for settled 128px squares.
    mask = (image[800:1000, 85:635].min(2) < 180).astype('uint8')
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    large = [row for row in stats[1:] if row[2] >= 100 and row[3] >= 100]
    if len(large) != 3 or any(not (126 <= row[2] <= 130 and 126 <= row[3] <= 130
            and row[4] >= row[2] * row[3] * .2) for row in large):
        raise ValueError('lesson exchange offer count or complete tile geometry is unresolved')
    boxes = sorted((tuple(map(int, row[:4])) for row in large))
    boxes = [(x + 85, y + 800, width, height) for x, y, width, height in boxes]
    if any(abs(box[0] - x) > 2 or abs(box[1] - 836) > 2
            for box, x in zip(boxes, (158, 296, 434), strict=True)):
        raise ValueError('lesson exchange offers do not fill the complete centered three-card row')
    return tuple(boxes)


def read_exchange_offers(reader, frames, *, registered_boxes=None, plan='sense', preferred=(), selected_index=None):
    from live_decision.inventory_customization import inspect_thumbnail_customizations
    if len(frames) != 1:
        raise ValueError('lesson exchange requires the current offer frame')
    if registered_boxes is not None:
        registered = tuple(tuple(box) for box in registered_boxes)
        if (len(registered) != 3 or any(len(box) != 4 or any(type(value) is not int for value in box)
                or abs(box[0] - x) > 2 or abs(box[1] - 836) > 2
                or not (126 <= box[2] <= 130 and 126 <= box[3] <= 130 and abs(box[2] - box[3]) <= 2)
                for box, x in zip(registered, (158, 296, 434), strict=True))):
            raise ValueError('lesson exchange original registered tile layout is invalid')
        # 163's selection corner touches the neighbouring tile's border.
        # Keep the already observed physical cells; every current identity,
        # badge, selected corner and enlarged title is still independently
        # read. The orange decoration cannot enlarge or redefine a tile.
        boxes = registered
    else:
        boxes = exchange_offer_boxes(frames[0])
    identities = reader.identify_batch(frames, boxes, plan=plan)
    from .candidate_pool import preferred_index

    target = selected_index if selected_index is not None else preferred_index(identities, preferred, 2)
    if identities[target]['card_id'] is None:
        raise ValueError('lesson exchange current offer identity is unresolved')
    # Unselected neighbours do not enter the ledger.
    known = [(index, row) for index, row in enumerate(identities) if row['card_id'] is not None]
    badges = dict(zip((index for index, _ in known), inspect_thumbnail_customizations(frames,
        [('exchange_offer', index, row['card_id'], row['box']) for index, row in known]), strict=True))
    if badges[target]['presence'] != 'CONFIDENT_ZERO':
        raise ValueError('lesson exchange offer permanent options require complete detail evidence')
    return [dict(index=index, card_id=row['card_id'], box=list(row['box']),
        customizations={} if badges.get(index, {}).get('presence') == 'CONFIDENT_ZERO' else None,
        identities=[asdict(proof) for proof in row['observations']], badge=badges.get(index, dict(presence='UNKNOWN')))
        for index, row in enumerate(identities)]


def selected_exchange_offer(parent, reader, frames, rows_by_frame, request, *, plan='sense'):
    image, = frames
    rows, = rows_by_frame
    chosen = request['offers'][request['selected_index']]
    expected = chosen['card_id']
    identity = reader.identify_batch(frames, [(408, 344, 144, 144)], plan=plan)[0]
    if identity['card_id'] != expected:
        raise ValueError('lesson exchange enlarged detail does not prove the original chosen card')
    name = ''.join(unicodedata.normalize('NFKC', parent.host.ledger.card_catalog.entries[expected].name).split())
    titles = [row for row in rows if row['score'] >= .95 and row['text'] == name
        and 100 <= row['box'][0] and 495 <= row['box'][1]
        and row['box'][0] + row['box'][2] <= 610 and row['box'][1] + row['box'][3] <= 565]
    if len(titles) != 1:
        raise ValueError('lesson exchange chosen-card detail title is unresolved')
    require_selection_corners(image, chosen['box'])
    button = exchange_offer_next(rows)
    if button is None:
        raise ValueError('lesson exchange next control is not uniquely observed')
    from .production import _common_caption_target
    return dict(card_id=expected, identities=[asdict(value) for value in identity['observations']],
        continue_target=_common_caption_target([dict(target=button['box'])]))


def require_selection_corners(image, box):
    x, y, width, height = box
    for left, top in ((x - 14, y - 14), (x + width - 3, y - 14),
            (x - 14, y + height - 3), (x + width - 3, y + height - 3)):
        corner = image[top:top + 17, left:left + 17]
        orange = (corner[:, :, 2] >= 220) & (corner[:, :, 1] >= 100) & (corner[:, :, 1] <= 220) & (corner[:, :, 0] <= 130)
        if orange.sum() < 30:
            raise ValueError('lesson exchange chosen tile lacks four complete selection corners')


def read_exchange_replacement(rows, image, *, day=1, catalog=None, request=None):
    from .home import _unique_home_template
    from .production import _row, _text

    if day not in (1, 4) or getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('replacement chooser requires its original lesson day and current image')
    # The original HIFExchangeFlag identifies the unselected panel. Once a
    # card is selected that panel is replaced by its name and artwork.
    try:
        _unique_home_template(image, 'produce/choose_move_cards.png', (0, 90, 720, 400), .8)
    except ValueError:
        pass
    else:
        return dict(scene='card_exchange', day=day, targets={})
    # fb8 deck.py ExchangeAuto verifies the chosen name and targets チェンジ.
    confirm = _row(rows, r'チ[ェエヱ]ンジ', (435, 1110, 190, 95))
    if catalog is not None:
        entry = catalog.entries.get((request or {}).get('outgoing_card', {}).get('card_id'))
        if entry is None or '基本' not in entry.name:
            raise ValueError('selected exchange source lacks its original catalog basic-card variant')
        name = ''.join(unicodedata.normalize('NFKC', entry.name).split())
        _text(rows, re.escape(name), (180, 250, 340, 65))
        # Both enlarged artworks and the selected original tile independently
        # prove the exact variant below. The title must retain its +, whereas
        # the old 646-only effect literal is not a basic-card family rule.
    else:
        _text(rows, r'パフォーマンスの基本', (180, 250, 340, 65))
    button = image[1125:1190, 380:620]
    orange = ((button[:, :, 2] >= 230) & (button[:, :, 1] >= 100)
        & (button[:, :, 1] <= 220) & (button[:, :, 0] <= 130)).mean()
    if orange < .65:
        raise ValueError('selected exchange source confirmation is not enabled')
    return dict(scene='card_exchange_selected', day=day, targets={'execute_lesson_exchange': tuple(confirm['box'])})


def replacement_chosen_identity(reader, frames, request, *, catalog=None, plan='sense'):
    identity = reader.identify_batch(frames, [(408, 96, 144, 144)], plan=plan)[0]
    expected = request['offers'][request['selected_index']]['card_id']
    if identity['card_id'] != expected and not _observed_upgraded_exchange(request, identity['card_id'], catalog):
        raise ValueError('replacement page does not retain the original chosen incoming card')
    return dict(card_id=identity['card_id'], identities=[asdict(value) for value in identity['observations']])


def _observed_upgraded_exchange(request, card_id, catalog):
    """Accept only an actually read + preview of this same original offer."""
    if catalog is None or request.get('outgoing_card', {}).get('upgraded') is not True:
        return False
    original = catalog.entries.get(request['offers'][request['selected_index']]['card_id'])
    current = catalog.entries.get(card_id)
    return (original is not None and current is not None and original.upgraded is False and current.upgraded is True
        and current.name == original.name + '+' and current.plan == original.plan and current.card_type == original.card_type)


def selected_incoming_id(request, catalog):
    original = request['offers'][request['selected_index']]['card_id']
    selected = request.get('selected_incoming')
    if selected is None:
        return original
    keys = selected.get('frame_ids', ())
    card_id = selected.get('card_id')
    if (selected.get('original_offer_card_id') != original or not isinstance(keys, (tuple, list)) or len(keys) != 1
            or not isinstance(keys[0], str) or not keys[0]
            or card_id != original and not _observed_upgraded_exchange(request, card_id, catalog)):
        raise ValueError('exchange incoming variant lacks its original observed selected preview')
    return card_id


def exchange_card_record(card):
    return dict(instance_id=card.instance_id, card_id=card.card_id, upgraded=card.upgraded,
        customization=[list(pair) for pair in card.customization])


def _nonbasic_marker_exclusion(reader, identity, state, basic_ids=frozenset({'646', '647'})):
    """Exclude another known art family without inventing its normal/+ ID."""
    from .ledger import PartialDeckLedgerState
    observations = identity.get('observations', ())
    gallery = getattr(getattr(reader, 'recognizer', None), 'gallery', None)
    if len(observations) != 1 or gallery is None:
        return None
    groups = {getattr(row, 'visual_group', None) for row in observations}
    if len(groups) != 1 or None in groups:
        return None
    group = next(iter(groups))
    members = {card_id for card_id, family in zip(gallery.card_ids, gallery.visual_group_ids, strict=True)
        if family == group}
    if (not members or members.intersection(basic_ids) or not isinstance(state, PartialDeckLedgerState)
            and not members.intersection(card.card_id for card in state.cards)):
        return None
    for row in observations:
        error, margin = row.reference_error, row.reference_margin
        if (row.reason != 'marker_identity_unresolved'
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in (error, margin))
                or not 0 <= error <= 20 or margin <= 0):
            return None
        if margin < 3:
            similarity, separation = row.embedding_similarity, row.embedding_margin
            if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (similarity, separation))
                    or not .85 <= similarity <= 1 or separation < .05):
                return None
    return dict(box=list(identity['box']), visual_group=group, card_id=None,
        source='independently_excluded_nonbasic_art_family', observations=[asdict(row) for row in observations])


def basic_source_group(parent, reader, frames, frame_ids, *, plan='sense'):
    from live_decision.inventory_customization import inspect_thumbnail_customizations

    from .ledger import CardInstance, PartialDeckLedgerState
    state = parent.host.ledger.state
    catalog = parent.host.ledger.card_catalog
    basic_ids = {key for key, entry in catalog.entries.items() if '基本' in entry.name and entry.plan in ('sense', 'free')}
    boxes = [(x, 638, 120, 120) for x in (80, 227, 373, 519)]
    identities = reader.identify_batch(frames, boxes, plan=plan)
    ids = [row['card_id'] for row in identities]
    partial = isinstance(state, PartialDeckLedgerState)
    if not partial and Counter(key for key in ids if key is not None) - Counter(card.card_id for card in state.cards):
        raise ValueError('replacement first-row identities disagree with the complete current inventory')
    first_basic = next((index for index, key in enumerate(ids) if key in basic_ids), None)
    if first_basic is None:
        raise ValueError('replacement plain-basic physical group is incomplete or ambiguous')
    chosen = ids[first_basic]
    exclusions = []
    for index, row in enumerate(identities):
        if row['card_id'] is None:
            proof = _nonbasic_marker_exclusion(reader, row, state, basic_ids)
            if proof is not None:
                exclusions.append(proof)
            elif index < first_basic:
                # An earlier unidentified tile could be the policy's first
                # basic. A later neighbor cannot outrank the verified target.
                raise ValueError('an earlier replacement tile could be the preferred basic card')
    indices = [index for index, key in enumerate(ids) if key == chosen]
    eligible = sorted((card for card in state.cards if card.card_id == chosen
        and card.upgraded == catalog.entries[chosen].upgraded and card.customization == ()), key=lambda card: card.instance_id)
    if not partial and len(indices) != len(eligible):
        raise ValueError('replacement plain-basic physical group is incomplete or ambiguous')
    badges = inspect_thumbnail_customizations(frames, [('exchange_source', index, chosen, boxes[index]) for index in indices])
    if any(row['presence'] != 'CONFIDENT_ZERO' for row in badges):
        raise ValueError('replacement basic source does not prove the original plain variant')
    if partial:
        # Only the visible copies of this verified plain variant are known.
        # Additional slots are evidence of inventory, not a grant of new cards.
        eligible = eligible[:len(indices)]
        for index in indices[len(eligible):]:
            eligible.append(CardInstance(f'{state.run_id}:exchange-visible:{state.day}:{index}',
                chosen, catalog.entries[chosen].upgraded))
        eligible.sort(key=lambda card: card.instance_id)
    mapping = [dict(index=index, box=list(boxes[index]), card=exchange_card_record(card),
        identity=[asdict(value) for value in identities[index]['observations']], badge=badge)
        for index, card, badge in zip(indices, eligible, badges, strict=True)]
    return dict(outgoing_card=mapping[0]['card'], outgoing_box=mapping[0]['box'],
        outgoing_group=mapping, outgoing_frame_ids=list(frame_ids),
        neighbor_family_exclusions=exclusions,
        outgoing_mapping_basis=('visible_identical_plain_group_display_ordinal' if partial
            else 'complete_identical_plain_group_display_ordinal'))


def exchange_selection_effect(host, effect):
    """Carry newly seen plain copies forward without changing the observed total."""
    from .ledger import CardInstance, CardObservationEffect
    request = (host.handoff or {}).get('nonbattle_context', {}).get('lesson_exchange', {})
    known = {card.instance_id for card in host.ledger.state.cards}
    fresh = []
    for row in request.get('outgoing_group', ()):
        card = row['card']
        if card['instance_id'] not in known:
            fresh.append(CardInstance(card['instance_id'], card['card_id'], card['upgraded']))
    if fresh:
        return CardObservationEffect(**vars(effect), observed_cards=tuple(fresh))
    return effect


def selected_source_identity(reader, frames, request, *, catalog=None, plan='sense'):
    expected = request.get('outgoing_card', {}).get('card_id')
    basic = expected == '646' if catalog is None else expected in catalog.entries and '基本' in catalog.entries[expected].name
    if not basic or not request.get('outgoing_group'):
        raise ValueError('selected replacement has no original plain-basic instance request')
    identities = reader.identify_batch(frames, [(169, 96, 144, 144), (35, 270, 144, 144)], plan=plan)
    if any(row['card_id'] != expected for row in identities):
        raise ValueError('selected replacement source detail changed identity')
    for image in frames:
        require_selection_corners(image, request['outgoing_box'])
    return dict(card_id=expected, identities=[[asdict(value) for value in row['observations']] for row in identities])


def lesson_exchange_effect(host):
    from .ledger import CardInstance, LedgerEffect, PartialDeckLedgerState
    state = host.ledger.state
    request = (host.handoff or {}).get('nonbattle_context', {}).get('lesson_exchange', {})
    matches = [card for card in state.cards if exchange_card_record(card) == request.get('outgoing_card')]
    group = request.get('outgoing_group') or []
    if (state.day not in (1, 4) or state.phase != 'card_exchange_selected' or len(matches) != 1
            or request.get('run_id') != state.run_id or request.get('game_pid') != host.game_pid
            or request.get('status') != 'active' or request.get('selected_index') not in range(3)
            or request.get('day', 1) != state.day
            or request.get('outgoing_mapping_basis') != ('visible_identical_plain_group_display_ordinal'
                if isinstance(state, PartialDeckLedgerState) else 'complete_identical_plain_group_display_ordinal')
            or not group or request.get('outgoing_card') != group[0]['card']
            or request.get('outgoing_box') != group[0]['box']):
        raise ValueError('lesson exchange requires its exact original selected physical instance')
    offer = request['offers'][request['selected_index']]
    entry = host.ledger.card_catalog.entries.get(selected_incoming_id(request, host.ledger.card_catalog))
    if entry is None or offer.get('customizations') != {}:
        raise ValueError('incoming lesson exchange card lacks complete original permanent options')
    incoming = CardInstance(f'{state.run_id}:exchange:{state.revision}:incoming', entry.card_id, entry.upgraded, ())
    return LedgerEffect(f'{state.run_id}:{state.revision}:execute-lesson-exchange', 'execute_lesson_exchange',
        remove_cards=(matches[0],), add_cards=(incoming,), phase='card_exchange_result')


def read_lesson_exchange_result(rows, catalog, request, *, day=1):
    from .calendar import read_route_resources
    from .production import _row
    from .nonbattle_observation import normalized, exchange_receipt_text
    if day not in (1, 4) or request.get('day', 1) != day:
        raise ValueError('lesson exchange result differs from its original actual day')
    expected = (request.get('outgoing_card', {}).get('card_id'), selected_incoming_id(request, catalog))
    import re
    message = re.fullmatch(r'(.+)を(.+)にチェンジしました[。]?', exchange_receipt_text(rows))
    outgoing, name = (normalized(catalog.entries[key].name) for key in expected)
    known_names = {normalized(entry.name) for entry in catalog.entries.values() if entry.plan in ('sense', 'free')}
    if (message is None or message[1] != outgoing
            or message[2] in known_names and message[2] != name):
        raise ValueError('lesson exchange result does not name the exact original outgoing and incoming cards')
    # The explicit result-card title proves incoming identity independently.
    # An unresolved duplicate name inside the message is not a second gate;
    # a different known card name in that message remains a contradiction.
    _row(rows, re.escape(name), (220, 775, 290, 65))
    resources = read_route_resources(rows)
    lines = [row for row in rows if row['score'] >= .95 and 80 <= row['box'][0] and 950 <= row['box'][1]
        and row['box'][0] + row['box'][2] <= 640 and row['box'][1] + row['box'][3] <= 1060]
    left, top = min(row['box'][0] for row in lines), min(row['box'][1] for row in lines)
    right = max(row['box'][0] + row['box'][2] for row in lines)
    bottom = max(row['box'][1] + row['box'][3] for row in lines)
    return dict(scene='card_exchange_result', day=day, exchange=list(expected),
        **resources,
        targets={'dismiss_exchange_result': (left, top, right - left, bottom - top)})
