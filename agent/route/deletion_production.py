"""Observe the native deletion menu; selection always names an owned instance."""
import re
import unicodedata
from dataclasses import asdict

import cv2
import numpy as np

from .deletion import DeletionRequest, request_record

KINDS = frozenset({'reward_delete_select', 'reward_delete_open', 'reward_delete_confirm'})
PICK_DESTINATIONS = ('reward_delete_selected', 'reward_delete_reselect')
SEARCH_DESTINATIONS = ('reward_delete_choice', 'reward_delete_search')


class OtherDeletionCardSelected(ValueError):
    """A known different card is displayed after selecting the original instance."""


class DeletionTargetNotFound(ValueError):
    """The current grid search has ended without the requested permanent card."""


def current_deletion_pick(parent, image, rows, key, value, shell):
    """Return a real selection, an explicitly missed free choice, or an unresolved read."""
    from .exchange import require_selection_corners

    if shell['kind'] == 'delete_selected':
        try:
            return bind_course_deletion(parent, [image], [rows], [key], value, shell)
        except OtherDeletionCardSelected:
            try:
                require_selection_corners(image, value['deletion']['box'])
            except ValueError:
                pass
            else:
                raise ValueError('current deletion detail conflicts with the selected target tile')
    # Re-find the original physical card in the current grid before any second tap.
    # Its box may move, but its owned instance and variant must still match.
    page = bind_course_deletion(parent, [image], [rows], [key], value, dict(shell, kind='delete_choice'))
    if page['kind'] == 'delete_search':
        raise ValueError('deletion reselection did not locate its original tile on the current page')
    return dict(page, kind='delete_reselect')


def _visible_deletion_boxes(image):
    # The native menu has four 120px cells per row at a 147px pitch.
    # Register their top and bottom borders on this viewport after any scroll.
    xs = (80, 227, 373, 519)
    edge = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 80, 180)
    border = np.stack([(edge[:, x+12:x+108] > 0).mean(axis=1) for x in xs])
    def row_alignment(top):
        starts = range(top, 991, 147)
        return sum(np.minimum(border[:, y-2:y+3].max(1), border[:, y+115:y+120].max(1)).mean()
            for y in starts) / len(starts)
    first = max(range(380, 527), key=row_alignment)
    return [(x, y, 120, 120) for y in range(first, 991, 147) for x in xs]


def _deletion_thumbnail(parent, image, retained):
    from .nonbattle_inventory import read_course_thumbnail, load_flat_inventory_readers
    reader = getattr(parent, 'deletion_reader', None)
    if reader is None:
        parent.deletion_reader = reader = load_flat_inventory_readers()[0]
    if retained is None:
        box = (80, 398, 120, 120)
        return box, read_course_thumbnail(reader, image, box)
    wanted = retained['request']['card']['card_id']
    visible = []
    for box in _visible_deletion_boxes(image):
        identity = read_course_thumbnail(reader, image, box)
        visible.append(identity['card_id'])
        if identity['card_id'] == wanted:
            return box, identity
    return None, visible


def _deletion_search(image, visible, retained, page):
    # hiro dialog_scroll reads the scrollbar at x=683 on the native deletion grid.
    center = image[380:1110, 679:688].astype('int16').max(axis=2)
    on = (center[:,4] >= 120) & (2*center[:,4] - center[:,0] - center[:,8] >= 50)
    segments = np.flatnonzero(np.diff(np.r_[False, on, False].astype('int8'))).reshape(-1, 2)
    thumbs = [(int(a), int(b)) for a,b in segments if b-a >= 40]
    if not thumbs:
        raise DeletionTargetNotFound('requested deletion card is absent from its unscrollable grid')
    top, _ = max(thumbs, key=lambda pair: pair[1]-pair[0])
    previous = retained.get('search') or {}
    direction = previous.get('direction', 'down' if top <= 15 else 'to_top')
    count = previous.get('steps', 0)
    if direction == 'to_top' and (top <= 15 or count >= 10):
        direction, count = 'down', 0
    elif direction == 'down' and (count >= 10 or previous.get('slots') == visible):
        direction, count = 'back', 0
    elif direction == 'back' and (count >= 10 or top <= 15 or previous.get('slots') == visible):
        raise DeletionTargetNotFound('requested deletion card was not found in the available pages')
    search = dict(direction=direction, steps=count+1, slots=visible)
    return dict(page, kind='delete_search', target_kind='reward_delete_scroll',
        target_key='reward_delete_scroll', target=(360,890 if direction == 'down' else 440,1,160),
        deletion=dict(retained, search=search))


def read_course_deletion_result(parent, image, rows, value):
    from .daily import complete_daily_resources
    from .calendar import read_calendar, read_route_resources
    from .production import _row

    retained = value.get('deletion')
    if retained is None:
        return None
    card_id = retained['request']['card']['card_id']
    name = ''.join(unicodedata.normalize('NFKC', parent.host._catalog.entries[card_id].name).split())
    text = name + 'を削除しました'
    if not any(row['score'] >= .95 and row['text'] == text for row in rows):
        return None
    target = _row(rows, re.escape(text), (60, 930, 600, 110))
    completed, crops = complete_daily_resources(parent.context, image, rows)
    day, _ = read_calendar(completed)
    if day != value['day']:
        raise ValueError('deletion result is not its original public-course calendar')
    resources = read_route_resources(completed)
    from .rewards import validate_resources
    validate_resources(parent.host, resources)
    return dict(resources, kind='delete_result', day=day, deletion=retained, result_text=text,
        resource_crops=crops, target=tuple(target['box']), target_key='reward_delete_result_ack',
        target_kind='reward_delete_result_ack')


def read_course_deletion_shell(image, rows, value, *, opened=False):
    from .production import _row

    retained = value.get('deletion') or {}
    last_menu = retained.get('last_menu', {}).get('kind')
    if last_menu == 'delete_confirm' or opened and last_menu == 'delete_selected':
        try:
            target = _row(rows, r'削除する|削除|決定|はい|OK', (360, 1080, 360, 160))
        except ValueError:
            pass
        else:
            if any('Pポイント' in row['text'] or 'ポイントを消費' in row['text'] for row in rows):
                raise ValueError('free event deletion cannot accept a priced confirmation')
            return dict(kind='delete_confirm', day=value['day'], delete_button=list(target['box']))
    if not any(row['text'] == '削除' and row['score'] >= .95 and row['box'][1] < 90 for row in rows):
        return None
    _row(rows, r'削除', (50, 20, 150, 70))
    consult_exit = value.get('source_target') == 'finish_consult'
    if not consult_exit and (value['special'] is not True or value.get('result', {}).get('special') is not True):
        raise ValueError('course deletion lacks its original acknowledged SP result')
    if consult_exit and any('キャンセル' in row['text'] or 'キヤンセル' in row['text']
            or 'Pポイント' in row['text'] for row in rows):
        raise ValueError('consultation exit event cannot enter an optional paid deletion menu')
    target = _row(rows, r'削除', (300, 1110, 150, 95))
    empty = any(row['text'] == 'スキルカードが選択されていません' and row['score'] >= .95 for row in rows)
    if empty:
        _row(rows, r'スキルカードが選択されていません', (120, 195, 480, 85))
    return dict(kind='delete_choice' if empty else 'delete_selected', day=value['day'],
        delete_button=list(target['box']))


def bind_course_deletion(parent, frames, rows_by_frame, frame_ids, value, page):
    from .exchange import require_selection_corners
    from .nonbattle_inventory import read_course_badge, read_course_thumbnail, load_flat_inventory_readers
    from .nonbattle_observation import _within

    if len(frames) != 1 or len(rows_by_frame) != 1 or len(frame_ids) != 1:
        raise ValueError('course deletion requires one current frame')
    host = parent.host
    retained = value.get('deletion')
    if retained is not None and retained.get('status') == 'committed_pending_census':
        raise ValueError('course deletion cannot remove a second card')
    if page['kind'] in ('delete_selected', 'delete_confirm'):
        if retained is None:
            raise ValueError('selected deletion card lacks its original request')
        matches = [card for card in host.ledger.pending.before.cards
            if request_record(DeletionRequest(card, 0)) == retained.get('request')]
        if len(matches) != 1 or not retained.get('box'):
            raise ValueError('deletion selected instance changed from its original request')
        box = tuple(retained['box'])
        card = matches[0]
        name = ''.join(unicodedata.normalize('NFKC', host._catalog.entries[card.card_id].name).split())
        if page['kind'] == 'delete_confirm':
            if (retained.get('last_menu', {}).get('kind') not in ('delete_selected', 'delete_confirm')
                    or value.get('source_target') != 'finish_consult' and not (
                        value.get('special') is True and value.get('result', {}).get('special') is True)):
                raise ValueError('free deletion confirmation lacks the verified selected instance')
            named = {''.join(unicodedata.normalize('NFKC', entry.name).split())
                for entry in host._catalog.entries.values()}
            for row in rows_by_frame[0]:
                displayed, verb, _ = row['text'].partition('を削除')
                displayed = displayed.strip('「」')
                if verb and displayed in named and displayed != name:
                    raise ValueError('deletion confirmation names another card')
            return dict(page, target=tuple(page['delete_button']), target_key='reward_delete_confirm',
                target_kind='reward_delete_confirm', deletion=retained, fee=retained['request']['fee'],
                fee_source='original_free_deletion_event')
        # Orange selection corners cover the thumbnail. Like the original
        # flow, re-read its enlarged exact name and current selected control.
        image = frames[0]
        known = {''.join(unicodedata.normalize('NFKC', entry.name).split()) for entry in host._catalog.entries.values()}
        titles = {row['text'] for row in rows_by_frame[0] if row['score'] >= .95
            and _within(row['box'], (150, 95, 510, 70)) and row['text'] in known}
        if len(titles) == 1 and name not in titles:
            raise OtherDeletionCardSelected('current deletion detail names another card')
        require_selection_corners(image, box)
        if titles != {name}:
            reader = getattr(parent, 'deletion_reader', None)
            if reader is None:
                parent.deletion_reader = reader = load_flat_inventory_readers()[0]
            if titles or read_course_thumbnail(reader, image, box)['card_id'] != card.card_id:
                raise ValueError('deletion selected card identity is unresolved')
        button = image[1125:1190, 280:450]
        orange = ((button[:, :, 2] >= 230) & (button[:, :, 1] >= 100)
            & (button[:, :, 1] <= 220) & (button[:, :, 0] <= 130)).mean()
        if orange < .65:
            raise ValueError('deletion selected-card button is not enabled')
        return dict(page, target=tuple(page['delete_button']), target_key='reward_delete_open',
            target_kind='reward_delete_open', deletion=retained)
    box, identity = _deletion_thumbnail(parent, frames[0], retained)
    if box is None:
        if (retained is None or retained.get('status') != 'active' or
                sum(request_record(DeletionRequest(card, 0)) == retained['request'] for card in host.ledger.pending.before.cards) != 1):
            raise ValueError('deletion grid search lacks the original permanent instance')
        return _deletion_search(frames[0], identity, retained, page)
    card_id = identity['card_id']
    matches = [card for card in host.ledger.pending.before.cards if card.card_id == card_id]
    if card_id is None or len(matches) != 1:
        raise ValueError('deletion first physical card is not a unique original inventory instance')
    card = matches[0]
    slot = ('deletion_first', 0, card_id, box)
    badge = read_course_badge(frames[0], slot)
    if card.customization or badge['presence'] != 'CONFIDENT_ZERO':
        raise ValueError('deletion first-card permanent options require complete detail evidence')
    request = request_record(DeletionRequest(card, 0))
    if retained is not None and retained.get('request') != request:
        raise ValueError('deletion first physical instance changed after its original selection')
    kind = 'reward_delete_select'
    return dict(page, target=box,
        target_key=kind, target_kind=kind,
        deletion=dict(retained or {}, schema_version=1, status='active', request=request, box=list(box),
            policy='D-065:hiro-v2.3.3:first-visible-card' if retained is None else retained.get('policy', 'existing_request'),
            source=('original_successful_consultation_exit' if value.get('source_target') == 'finish_consult'
                else 'original_successful_SP_course_result_acknowledgement'),
            frame_ids=list(frame_ids), identities=[asdict(row) for row in identity['observations']], badge=badge))
