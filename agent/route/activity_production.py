"""Current D3 outing and D6 free consultation rewards in the normal route host.

One original reservation stays open until every reward and the fresh physical
inventory are independently observed. This producer never issues input itself.
"""
from __future__ import annotations

import re
import csv
import time
import unicodedata
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from utils.runtime_paths import resolve_runtime_path

from . import activity_reward
from .hif import active_hif_role
from .host import HifRoutePage
from .engine import Target, Outcome, Decision, Observation, propose_unique

DESTINATIONS = {
    'activity_ack_support_card': ('support_reward', 'support_lesson', 'support_ok', 'activity_home', 'consult_receipt', 'consult_shop'),
    'activity_ack_support_ok': ('support_reward', 'support_lesson', 'support_ok', 'activity_home', 'consult_receipt', 'consult_shop'),
    'select_outing': ('outing_selected',), 'confirm_outing': ('outing_choice',),
    'preview_outing_first': ('outing_choice_preview',), 'execute_outing': ('drink_offer',),
    'activity_select_drink': ('drink_selected',), 'activity_receive_drink': ('drink_receipt',),
    'activity_ack_drink': ('card_offer',), 'activity_select_card': ('card_selected',),
    'activity_receive_card': ('card_receipt',), 'activity_ack_card': ('card_offer', 'activity_home'),
    'activity_inspect_card': ('card_inspection',), 'activity_inspect_drink': ('drink_inspection',),
    'select_consult': ('consult_selected',), 'confirm_consult': ('consult_receipt', 'consult_shop'),
    'activity_ack_free_drink': ('consult_receipt', 'consult_shop'), 'finish_consult': ('battle_intro', 'reward_delete_choice'),
    'recovery_inventory': ('activity_inventory',), 'recovery_inventory_detail': ('activity_inventory',),
    'recovery_inventory_scroll': ('activity_inventory',), 'recovery_close': ('activity_home', 'consult_shop', 'support_lesson'),
    'activity_open_drink_detail': ('activity_drink_detail',),
    'activity_close_drink_detail': ('activity_home', 'consult_shop'),
}


# hiro quick_page returns this blank point once ACQUIRE identifies the award.
ACQUIRE_SKIP = (12, 12, 1, 1)


def canonical(text):
    return ''.join(unicodedata.normalize('NFKC', text).split())


def drink_catalog():
    path = resolve_runtime_path(Path('assets/data/p_drinks.csv'), Path('data/p_drinks.csv'))
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    result = {canonical(row['name']): row['id'] for row in rows}
    if len(result) != len(rows): raise ValueError('drink catalog names are ambiguous')
    return result


def receipt_panel(image):
    """Recognise an award band when its selected title is temporarily weak."""
    if getattr(image, 'shape', ()) != (1280, 720, 3):
        raise ValueError('activity receipt capture is uncalibrated')
    # Upstream acknowledges the obtained page, not its border raster. Keep
    # the existing positive band family for provisional acknowledgements;
    # do not audit continuous edges on an already received free reward.
    patches = [cv2.cvtColor(image[850:858, x:x+25], cv2.COLOR_BGR2HSV) for x in (100, 580)]
    styles = (((95, 115), (95, 115)), ((15, 30), (15, 30)), ((15, 30), (75, 95)))
    if not any(all(((hsv[:, :, 0] >= low) & (hsv[:, :, 0] <= high)
            & (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 180)).mean() >= .85
            for hsv, (low, high) in zip(patches, style, strict=True)) for style in styles):
        raise ValueError('free activity receipt title band is incomplete')


def obtained_receipt_visible(image):
    try:
        receipt_panel(image)
    except ValueError:
        return False
    return True


def receipt_title(context, image, rows, names, *, minimum=.95):
    from .production import _row
    receipt_panel(image)
    pattern = '|'.join(re.escape(name) for name in names)
    try:
        return _row(rows, pattern, (150, 835, 470, 65), minimum=minimum)
    except ValueError:
        from live_decision.hif_recovery import _field
        # A weak full-frame label is not promoted. Read the complete current
        # band independently with its original .95 acceptance requirement.
        return _field(context, image, 'HifActivityFreeReceiptTitle', (150, 835, 470, 65), pattern)


def consult_drink_title(context, image, rows, names):
    """Read a randomly granted drink, including titles joined to the cup icon."""
    from live_decision.hif_recovery import _field

    from .nonbattle_observation import _within
    try:
        return receipt_title(context, image, rows, names)
    except ValueError:
        receipt_panel(image)
    labels = [row for row in rows if _within(row['box'], (150, 835, 470, 65))]
    if len(labels) != 1:
        raise ValueError('consult obtained drink has no unique current title band label')
    x, y, w, h = labels[0]['box']
    ink = (image[y:y+h, x:x+w].max(axis=2) < 150).any(axis=0)
    # The leading cup and the first character are separated by a blank gap.
    # Its width follows the actual line. Read all remaining characters anew.
    gap = next((i for i in range(h//2, min(h+1, w-4))
        if ink[i-1] and not ink[i:i+4].any()), None)
    if gap is None:
        raise ValueError('consult obtained drink name cannot be separated from its icon')
    crop = (x+gap-2, y-2, w-gap+4, h+1)
    pattern = '|'.join(re.escape(name) for name in names)
    return _field(context, image, 'HifConsultDrinkName', crop, pattern)


def selected_award_receipt(context, image, rows, name):
    """Acknowledge an obtained panel without inventing the awarded identity.

    Only the original selected/received transaction may call this. If the
    title is unreadable, its expected identity stays provisional until the
    final inventory census (and drink slot detail) independently agrees.
    """
    try:
        return dict(receipt_title(context, image, rows, (name,)),
            receipt_identity_source='current_title')
    except ValueError:
        receipt_panel(image)
        # This complete band is the independently recognized award control,
        # not a fallback choice tile or a new receive/consume button.
        return dict(box=(75, 844, 570, 60),
            receipt_identity_source='pending_selection_requires_census')


def support_card_award(image, rows, catalog):
    """Positive obtained-card family; a title supplies an expected ID only."""
    try:
        receipt_panel(image)
    except ValueError:
        return None
    names = {}
    for identity, entry in catalog.entries.items():
        names.setdefault(canonical(entry.name), []).append(identity)
    titles = [row for row in rows if row['score'] >= .95 and 150 <= row['box'][0]
        and 835 <= row['box'][1] and row['box'][0]+row['box'][2] <= 620
        and row['box'][1]+row['box'][3] <= 900 and canonical(row['text']) in names]
    if len(titles) != 1 or len(names[canonical(titles[0]['text'])]) != 1:
        raise ValueError('support obtained-card identity is not uniquely observed')
    identity = names[canonical(titles[0]['text'])][0]
    return dict(card_id=identity, upgraded=catalog.entries[identity].upgraded)


def support_award(image, rows, host, *, upgrade_only=False):
    """Read an automatic upgrade or an obtained card before free acknowledgement."""
    from .ledger import PartialDeckLedgerState
    from .production import _row
    from .nonbattle_observation import exchange_receipt_text
    message = re.fullmatch(r'(.+)を強化しました[。]?', exchange_receipt_text(rows))
    if message is not None:
        originals = [card for card in host.ledger.state.cards if not card.upgraded
            and canonical(host._catalog.entries[card.card_id].name) == message[1]]
        upgraded = [entry for entry in host._catalog.entries.values() if entry.upgraded
            and canonical(entry.name) == message[1] + '+']
        if isinstance(host.ledger.state, PartialDeckLedgerState):
            variants = [card for card in host._catalog.entries.values() if not card.upgraded
                and card.plan in ('sense', 'free') and canonical(card.name) == message[1]]
            if len(variants) != 1 or len(upgraded) != 1:
                raise ValueError('support upgrade has no unique base and upgraded variant')
            source_id, instance = variants[0].card_id, None
        else:
            if len(originals) != 1 or originals[0].customization or len(upgraded) != 1:
                raise ValueError('support upgrade has no unique original permanent instance and variant')
            source_id, instance = originals[0].card_id, originals[0].instance_id
        entry = upgraded[0]
        _row(rows, re.escape(canonical(entry.name)), (220, 775, 290, 65))
        lines = [row for row in rows if row['score'] >= .95 and 80 <= row['box'][0] and 950 <= row['box'][1]
            and row['box'][0] + row['box'][2] <= 640 and row['box'][1] + row['box'][3] <= 1060]
        left, top = min(row['box'][0] for row in lines), min(row['box'][1] for row in lines)
        right = max(row['box'][0] + row['box'][2] for row in lines)
        bottom = max(row['box'][1] + row['box'][3] for row in lines)
        return dict(grant=dict(kind='upgrade', source_instance_id=instance,
            source_card_id=source_id, card_id=entry.card_id, upgraded=True),
            target=(left, top, right-left, bottom-top))
    if upgrade_only:
        return None
    grant = support_card_award(image, rows, host._catalog)
    return dict(grant=grant, target=(75, 844, 570, 60)) if grant is not None else None


def support_lesson_inventory_button(context, image, client_size):
    from types import SimpleNamespace

    from .production import recognize_route_inventory_button as recognize_inventory_button

    def recognize(node, picture, *, pipeline_override):
        config = pipeline_override[node]
        # The lesson footer includes Fast Forward, moving this same bag glyph
        # left of the world/Live ROI. Keep its assets/processing/threshold and
        # current-frame boxes; only select this known page's complete region.
        return context.run_recognition(node, picture, pipeline_override={node: dict(config, roi=[470, 1150, 100, 100])})
    return recognize_inventory_button(SimpleNamespace(run_recognition=recognize), image, client_size)


def enabled_receive(image, rows):
    from .production import _row
    label = _row(rows, r'受け取る', (270, 1060, 180, 80))
    patch = image[1065:1135, 235:485]
    orange = (patch[:, :, 2] >= 220) & (patch[:, :, 1] >= 100) & (patch[:, :, 1] <= 220) & (patch[:, :, 0] <= 140)
    if orange.mean() < .65: raise ValueError('activity selected reward receive button is disabled')
    return tuple(label['box'])


def _retry_reward_upgrade(reader, image, box, observed):
    """Retry a known reward-card family with the adjacent effect-icon area masked."""
    if observed.reason != 'marker_identity_unresolved' or observed.marker_errors is None:
        return None
    recognizer = reader.recognizer
    normal, upgraded = observed.marker_errors
    maximum = recognizer.upgrade_maximum_distance
    margin = recognizer.upgrade_minimum_margin
    if upgraded <= maximum or normal - upgraded < margin:
        return None

    from PIL import Image
    from card_selection import EmbeddingHit, resolve_card_identity, extract_upgrade_marker

    x, y, width, height = box
    crop = image[y:y + height, x:x + width]
    if recognizer.embedder.source_color_order == 'BGR':
        crop = crop[:, :, ::-1]
    query = extract_upgrade_marker(Image.fromarray(np.ascontiguousarray(crop)))

    # Use the same 16px templates and error thresholds as the full marker read.
    # The lower-left corner can contain a reward effect icon beside the '+'.
    mask = np.ones((16, 16), dtype=np.uint8)
    mask[-5:, :5] = 0
    pixel_count = int(mask.sum()) * query.shape[2]
    gallery = recognizer.gallery
    scores = [float('inf'), float('inf')]
    members = set()
    for index in reader.groups[observed.visual_group]:
        variant = gallery.upgrade_counts[index]
        template = gallery.upgrade_markers[index][4:20, 4:20].copy()
        error = cv2.minMaxLoc(cv2.matchTemplate(query, template, cv2.TM_SQDIFF, mask=mask))[0]
        scores[variant] = min(scores[variant], error / pixel_count)
        members.add((gallery.card_ids[index], variant))
    if scores[1] > maximum or scores[0] - scores[1] < margin:
        return None
    card_ids, upgrades = zip(*sorted(members), strict=True)
    hit = EmbeddingHit(card_ids[0], 'inventory', 1.0, 0, observed.visual_group, card_ids, upgrades)
    try:
        return resolve_card_identity((hit,), plan='sense', upgraded=True).card_id
    except ValueError:
        return None


def read_card_offers(reader, frames, catalog, *, registered=None, focus=None):
    from .outing import outing_offer_boxes
    from .nonbattle_inventory import read_course_thumbnail
    if len(frames) != 1: raise ValueError('activity offers require one current frame')
    if registered is None:
        boxes = outing_offer_boxes(frames[0])
    else:
        boxes = tuple(tuple(row['box']) for row in registered)
        if len(boxes) != 3 or any(len(box) != 4 or any(type(v) is not int for v in box)
                or abs(box[0]-x) > 2 or abs(box[1]-821) > 2 or not 126 <= box[2] <= 130
                or not 126 <= box[3] <= 130 for box, x in zip(boxes, (158, 296, 434), strict=True)):
            raise ValueError('activity original physical offers are invalid')
    offers = []
    for index, box in enumerate(boxes):
        if focus is not None and index != focus:
            offers.append(deepcopy(registered[index]))
            continue
        row = read_course_thumbnail(reader, frames[0], box)
        card_id, extra = row.get('card_id'), {}
        observed = row['observations'][0]
        if card_id is None and registered is None:
            card_id = _retry_reward_upgrade(reader, frames[0], box, observed)
        if card_id not in catalog.entries:
            card_id = None
        previous = registered[index] if registered is not None else {}
        if card_id is None or previous.get('candidate_ids') or previous.get('resolved_by_detail'):
            group = getattr(observed, 'visual_group', None)
            if group is not None and observed.reason != 'reference_group_missing':
                gallery = reader.recognizer.gallery
                candidates = sorted({gallery.card_ids[i] for i in reader.groups.get(group, ())
                    if gallery.card_ids[i] in catalog.entries and catalog.entries[gallery.card_ids[i]].plan in ('sense', 'free')})
                if candidates:
                    extra = dict(visual_group=group, candidate_ids=candidates)
            if previous.get('candidate_ids'):
                if extra and extra['candidate_ids'] != previous['candidate_ids']:
                    raise ValueError('activity inspected card artwork changed')
                if group is not None and group != previous.get('visual_group'):
                    raise ValueError('activity inspected card artwork changed')
                extra = dict(visual_group=previous['visual_group'], candidate_ids=previous['candidate_ids'])
            if previous.get('resolved_by_detail'):
                if card_id is not None and previous['card_id'] != card_id:
                    raise ValueError('activity inspected card artwork changed')
                card_id = previous['card_id']
                extra['resolved_by_detail'] = True
        offers.append(dict(index=index, card_id=card_id,
            upgraded=catalog.entries[card_id].upgraded if card_id is not None else None,
            box=list(box), customizations=None, **extra))
    # Persist business data and original physical cells, not fluctuating scores.
    return offers


def decide_activity_reward(host, observation, config):
    page = host._page
    if page is None or page.observation != observation or not isinstance(page.activity, dict):
        return Decision(Outcome.STOP, 'hif_activity_current_complete_page_required')
    value = activity_reward.request(host)
    if value is not None and value['stage'] == 'census' and not observation.targets:
        try:
            host.bind_observed_activity_reward(page)
        except ValueError as error:
            return Decision(Outcome.STOP, 'hif_activity_census_rejected: ' + str(error))
        if activity_reward.observed_lesson_grants(value):
            return Decision(Outcome.PAUSE, 'hif_observed_lesson_support_grant_complete')
        return Decision(Outcome.PAUSE, 'hif_observed_activity_reward_census_complete')
    if len(observation.targets) != 1:
        return Decision(Outcome.STOP, 'hif_activity_unique_supported_action_required')
    kind = observation.targets[0].kind
    if kind not in DESTINATIONS:
        return Decision(Outcome.STOP, 'hif_activity_action_not_bound')
    if kind in ('activity_select_drink', 'activity_receive_drink') and all(
            row['state'] == 'OCCUPIED' for row in page.activity.get('drink_slots', ())):
        return Decision(Outcome.STOP, 'hif_activity_full_drinks_require_verified_decline')
    destinations = DESTINATIONS[kind]
    if kind == 'activity_ack_card' and value is not None and activity_reward.outing_rewards_received(value):
        destinations = (*destinations, 'support_reward')
    if kind == 'select_consult' and any(effect.target_key == kind and effect.phase == 'consult_rewards' for effect in page.effects):
        destinations = ('consult_receipt','consult_shop','support_reward','support_ok')
    return propose_unique(observation, observation.targets, 'hiro-current-free-activity-reward', destinations)


def activity_page_id(day, kind, transaction_id, page):
    inventory = page.get('inventory') or {}
    index = page.get('selected_offer_index', page.get('inspect_offer'))
    offers = page.get('offers', ())
    offer = offers[index] if type(index) is int and index in range(len(offers)) else {}
    # A one-pixel registration difference does not change a physical inventory page.
    inventory_view = ((inventory.get('total'), tuple(sorted(map(int, inventory.get('visible_boxes', {})))))
        if kind == 'activity_inventory' else inventory.get('displacement'))
    identity = (f'activity:{day}:{kind}:{transaction_id}:{inventory_view}:{inventory.get("selected_index")}:'
        f'{sorted(page.get("chosen", {}).items())}:{offer.get("card_id", offer.get("drink_id"))}:'
        f'{page.get("inspected", {}).get("index")}:{index}:'
        f'{[(row["slot"], row["state"]) for row in (page.get("drink_slots") or ())]}')
    if kind == 'support_reward':
        identity += ':' + str(sorted(page['grant'].items()))
    return identity


class HifActivityProducer:
    def __init__(self, parent):
        self.parent = parent
        self.inventory = None
        self.inventory_scope = None
        self.offer_reader = None
        self.names = drink_catalog()

    def _slots(self, image, *, offset, maintain_empty=False, quoted_count=None):
        from .production import read_current_drink_occupancy, maintain_empty_nonbattle_drinks
        if quoted_count is not None:
            from .production import read_quoted_nonbattle_drinks
            state = self.parent.host.ledger.state
            return read_quoted_nonbattle_drinks(self.parent.context, image, state.drinks,
                footer_offset=offset, revision=state.revision, expected_count=quoted_count)
        if maintain_empty:
            state = self.parent.host.ledger.state
            return maintain_empty_nonbattle_drinks(self.parent.context, image, state.drinks,
                footer_offset=offset, revision=state.revision)
        return read_current_drink_occupancy(self.parent.context, image, footer_offset=offset)

    def _inventory(self, rows, value):
        from .production import read_nonbattle_inventory
        page = read_nonbattle_inventory(rows)
        if page['phase'] != 'inventory' or value is None or value['stage'] != 'census':
            raise ValueError('activity owned-card page lacks its original census reservation')
        from .nonbattle_inventory import bind_inventory_census
        saved = bind_inventory_census(self, ['activity', value['transaction_id']], value.get('inventory'))
        if self.inventory is None:
            from .nonbattle_inventory import FlatInventoryCollector, load_flat_inventory_readers
            collector = FlatInventoryCollector(*load_flat_inventory_readers())
            collector.census_session = self.inventory_scope
            if saved is not None:
                if self.parent.recovery_report is None and not saved.get('durable_frames'):
                    raise ValueError('partial activity census needs its original source registration')
                self.parent._restore_inventory(collector=collector, retained=saved, phase='activity_inventory')
                if collector.previous is None:
                    raise ValueError('partial activity census registration was not restored')
            self.inventory = collector
        return dict(kind='activity_inventory', day=value['census_day'], target=page['target'],
            target_key='recovery_close', target_kind='recovery_close')

    def _read(self, image, rows, *, calendar=None):
        from .daily import complete_daily_resources
        from .consult import read_consult_shop
        from .calendar import read_calendar, read_route_resources
        from .production import _row
        from .nonbattle_observation import _within
        host = self.parent.host
        value = activity_reward.request(host)
        gate = host.hooks.transition_gate
        pending = gate.pending
        if any(row['text'] in ('所持スキルカード', '所持Pアイテム') and row['score'] >= .95 for row in rows):
            return self._inventory(rows, value)
        outing_upgrade = (value is not None and value['kind'] == 'outing'
            and value['stage'] in ('card_receipt', 'support_reward', 'support_ok')
            and activity_reward.outing_rewards_received(value))
        completed = None
        if value is not None and (outing_upgrade or value['kind'] == 'support'
                or value['kind'] == 'consult' and value['stage'] != 'census'):
            completed, crops = complete_daily_resources(self.parent.context, image, rows)
            resources = read_route_resources(completed)
            try:
                award = support_award(image, completed, host, upgrade_only=outing_upgrade)
            except ValueError:
                if value['kind'] != 'consult':
                    raise
                # A drink receipt uses this same panel. Its own reader below
                # must prove its identity; an unknown card yields no target.
                award = None
            if award is not None:
                return dict(resources, kind='support_reward', day=value['day'], grant=award['grant'],
                    target=award['target'], target_key='activity_ack_support_card', target_kind='activity_ack_support_card')
            if (not outing_upgrade or value['stage'] != 'card_receipt') and any(
                    row['score'] >= .95 and row['text'] == 'サポートイベント効果'
                    and 10 <= row['box'][0] and 960 <= row['box'][1]
                    and row['box'][0]+row['box'][2] <= 630 and row['box'][1]+row['box'][3] <= 1020 for row in completed):
                button = _row(completed, r'OK', (250, 1080, 260, 140))
                return dict(resources, kind='support_ok', day=value['day'], target=tuple(button['box']),
                    target_key='activity_ack_support_ok', target_kind='activity_ack_support_ok')
        if value is not None and value['kind'] == 'support':
            if value.get('destination_scene') == 'activity_home':
                from .footer import read_footer_offset
                day, _ = read_calendar(completed)
                if day != value['census_day']:
                    raise ValueError('support effects have not reached their actual next day')
                offset, footer = read_footer_offset(image, context=self.parent.context)
                return dict(resources, kind='activity_home', day=day, resource_crops=crops,
                    footer_evidence=footer, drink_slots=self._slots(image, offset=offset,
                        quoted_count=len(host.ledger.pending.before.drinks)))
            from .daily import read_lesson_options
            page = read_lesson_options(image, completed, day=value['day'])
            if page['scene'] != 'lesson_choice':
                raise ValueError('support grant must return to its unselected lesson choices')
            if activity_reward.observed_lesson_grants(value):
                return dict(resources, kind='support_lesson', day=value['day'], resource_crops=crops)
            return dict(resources, kind='support_lesson', day=value['day'], resource_crops=crops,
                drink_slots=self._slots(image, offset=0, quoted_count=len(host.ledger.state.drinks)))
        if value is not None and (value.get('inspecting_slot') is not None or
                pending is not None and pending.target.kind == 'activity_open_drink_detail'):
            if any(row['text'] == 'Pドリンク詳細' and row['score'] >= .85 for row in rows):
                from .recovery_drinks import read_detail
                detail = read_detail(self.parent.context, image, rows, self.names)
                return dict(kind='activity_drink_detail', day=value['census_day'],
                    inspect_slot=value['inspecting_slot'], drink_id=self.names[detail['name']],
                    target=detail['target'], target_key='activity_close_drink_detail', target_kind='activity_close_drink_detail')
        if pending is not None and pending.target.kind == 'finish_consult':
            from .rewards import consult_exit_request
            from .deletion_production import read_course_deletion_shell
            deletion = read_course_deletion_shell(image, rows, consult_exit_request(host, gate))
            if deletion is not None:
                if deletion['kind'] != 'delete_choice':
                    raise ValueError('consultation exit cannot adopt an already selected deletion')
                return deletion
            from .supply import read_hif_battle_introduction
            page = read_hif_battle_introduction(rows, context=self.parent.context, image=image)
            # This exact original shop exit is zero resource; the independent
            # ranking introduction hides, rather than re-observes, the HUD.
            state = host.ledger.state
            return dict(kind='battle_intro', day=7, stamina=state.stamina,
                max_stamina=state.max_stamina, currency=state.currency,
                maintained_resource_revision=state.revision)
        if completed is None:
            completed, crops = complete_daily_resources(self.parent.context, image, rows)
            resources = read_route_resources(completed)
        foreground_activity = any(row['score'] >= .95 and row['text'] in ('相談','おでかけ') and row['box'][1] < 100
            for row in completed)
        if not foreground_activity and (calendar is not None or any(
                row['text'] == 'H.I.F本戦まで' and row['score'] >= .95 for row in completed)):
            calendar = calendar or read_calendar(completed)
            day, family = calendar
            from .footer import read_footer_offset
            offset, footer = read_footer_offset(image, context=self.parent.context)
            selected = any(row['text'] == 'SELECT' and row['score'] >= .95 for row in completed)
            if value is not None:
                if value['kind'] == 'consult' and value['stage'] == 'paid_pending' and day == 6 and family == 'consult_entry':
                    from .consult import read_consult_schedule
                    page = read_consult_schedule(completed, selected=selected, calendar=calendar)
                    page.update(footer_offset=offset,footer_evidence=footer,resource_crops=crops,
                        drink_slots=self._slots(image,offset=offset,quoted_count=len(host.ledger.pending.before.drinks)))
                    return page
                if value['kind'] != 'outing' or day != 4:
                    raise ValueError('activity calendar is not its actual next day')
                page = dict(resources, kind='activity_home', day=day)
            elif day == 3 and family == 'outing':
                from .outing import read_outing_schedule
                page = read_outing_schedule(completed, selected=selected, calendar=calendar)
                page['kind'] = page.pop('scene')
            elif day == 6 and family == 'consult_entry':
                from .consult import read_consult_schedule
                page = read_consult_schedule(completed, selected=selected, calendar=calendar)
            else:
                raise ValueError('activity requires the actual original outing/consult calendar')
            page.update(footer_offset=offset, footer_evidence=footer,
                drink_slots=self._slots(image, offset=offset,
                    maintain_empty=value is None and not host.ledger.state.drinks,
                    quoted_count=(len(host.ledger.state.drinks) + len(value['drinks']) if value is not None else
                        len(host.ledger.state.drinks) if host.ledger.state.drinks else None)), resource_crops=crops)
            return page
        outing = any(row['text'] == 'おでかけ' and row['score'] >= .95 and row['box'][1] < 100 for row in completed)
        consult = any(row['text'] == '相談' and row['score'] >= .95 and row['box'][1] < 100 for row in completed)
        if not outing and not consult: raise ValueError('activity enclosing outing/consultation page is unknown')
        if value is None:
            if consult:
                activity_reward.completed_consult(host)
                from .footer import read_footer_offset
                from .production import read_bound_drink_slots
                offset, footer = read_footer_offset(image, context=self.parent.context)
                page = read_consult_shop(completed, image)
                page['quote'] = deepcopy(activity_reward.completed_consult(host)['resolution']['quote'])
                page['drink_slots'] = read_bound_drink_slots(self.parent.context, image, host.ledger.state.drinks,
                    footer_offset=offset, maintained_revision=host.ledger.state.revision)
                page.update(footer_offset=offset, footer_evidence=footer)
                if (page['stamina'], page['max_stamina'], page['currency']) != (
                        host.ledger.state.stamina, host.ledger.state.max_stamina, host.ledger.state.currency):
                    raise ValueError('completed consultation shop resources changed from the settled activity')
                return page
            else:
                from .outing import read_outing_options
                selected = any('最大体力の60%' in row['text'] and row['score'] >= .95 for row in completed)
                page = read_outing_options(completed, selected=selected)
                page['kind'] = page.pop('scene')
        else:
            if outing != (value['kind'] == 'outing'): raise ValueError('activity source family changed')
            stage = value['stage']
            # A child confirmation reads the destination, not its old source.
            action = pending.target.kind if pending is not None else None
            receipt_expected = (consult and stage in ('paid_pending', 'support_reward', 'support_ok', 'consult_receipt') or
                action in ('activity_receive_card', 'activity_receive_drink') or stage in ('card_receipt', 'drink_receipt'))
            try:
                if not receipt_expected: raise ValueError('activity receipt is outside its original stage')
                if consult:
                    title = consult_drink_title(self.parent.context, image, completed, self.names)
                    chosen = dict(drink_id=self.names[title['text']])
                    kind, key = 'consult_receipt', 'activity_ack_free_drink'
                else:
                    chosen = value.get('chosen')
                    if not isinstance(chosen, dict): raise ValueError('activity receipt has no original selected reward')
                    card = 'card_id' in chosen
                    name = canonical(host._catalog.entries[chosen['card_id']].name) if card else next(name for name, identity in self.names.items() if identity == chosen['drink_id'])
                    title = selected_award_receipt(self.parent.context, image, completed, name)
                    kind, key = ('card_receipt', 'activity_ack_card') if card else ('drink_receipt', 'activity_ack_drink')
                page = dict(resources, day=value['day'], kind=kind, chosen=deepcopy(chosen),
                    target=tuple(title['box']), target_key=key, target_kind=key)
                if not consult:
                    page['receipt_identity_source'] = title['receipt_identity_source']
                    # This zero-cost animation acknowledgement is neither a
                    # choice page nor a current drink-slot census.
                    return page
            except ValueError:
                if consult:
                    from .consult import read_consult_reward_quote
                    if obtained_receipt_visible(image):
                        raise
                    try:
                        quote = read_consult_reward_quote(completed)
                    except ValueError:
                        page = read_consult_shop(completed, image)
                        page['quote'] = deepcopy((value.get('consult_reward_quote') or {}).get('quote', {}))
                    else:
                        if action != 'activity_ack_free_drink':
                            raise ValueError('consult support/P overlay has no original submitted free acknowledgement')
                        page = dict(resources, day=6, kind='consult_reward_transition', quote=quote)
                else:
                    # fb8 HIFDrinkFlag and deck.py _is_card_screen use the
                    # reward family in these regions. The remaining words
                    # only instruct the player to choose an offer.
                    drink_prompt = any('受け取るPドリンク' in row['text'] and _within(row['box'], (80, 530, 560, 200))
                        for row in completed)
                    card_prompt = any('受け取るスキルカード' in row['text'] and _within(row['box'], (90, 570, 560, 110))
                        for row in completed)
                    if drink_prompt and card_prompt:
                        raise ValueError('conflicting activity reward families')
                    selected_drink = stage == 'drink_selected' or action == 'activity_select_drink'
                    inspecting_drink = action == 'activity_inspect_drink' or stage == 'drink_inspection' and action != 'activity_select_drink'
                    inspecting = action == 'activity_inspect_card' or stage == 'card_inspection' and action != 'activity_select_card'
                    selected_card = stage == 'card_selected' or action == 'activity_select_card'
                    if not any((drink_prompt, card_prompt, selected_drink, selected_card, inspecting, inspecting_drink)):
                        raise ValueError('activity free reward choice is not completely observed')
                    if drink_prompt or selected_drink or inspecting_drink:
                        kind = 'drink_inspection' if inspecting_drink else 'drink_selected' if selected_drink else 'drink_offer'
                        key = 'activity_receive_drink' if selected_drink else 'activity_select_drink'
                    else:
                        kind = 'card_inspection' if inspecting else 'card_selected' if selected_card else 'card_offer'
                        key = 'activity_receive_card' if selected_card else 'activity_select_card'
                    page = dict(resources, kind=kind, day=3, target_key=key, target_kind=key)
        if outing:
            offset = 0
        elif page['kind'] == 'consult_shop':
            from .footer import read_footer_offset
            offset, footer = read_footer_offset(image, context=self.parent.context)
            page['footer_evidence'] = footer
        else:
            # The retained obtained/support overlay has its own calibrated
            # +13 footer and can cover the menu. Only the unobstructed shop
            # needs to locate its current footer.
            offset = 13
        page.update(drink_slots=self._slots(image, offset=offset,
            maintain_empty=value is None and not host.ledger.state.drinks), footer_offset=offset, resource_crops=crops)
        return page

    def select_offer_index(self, offers):
        """Use the configured card preference without changing drink selection."""
        if all('card_id' in offer for offer in offers):
            from .candidate_pool import candidate_policy, preferred_index

            preferred, _ = candidate_policy(self.parent.context)
            return preferred_index(offers, preferred, offers[0]['index'])
        return offers[0]['index']

    def _complete_offer(self, pages, images, rows):
        from .activity_offers import complete_offer
        page, = pages
        image, = images
        text, = rows
        complete_offer(self, page, image, text)

    def _complete_census(self, pages, images, rows, keys):
        parent, host = self.parent, self.parent.host
        value = activity_reward.request(host)
        page, gate = pages[-1], host.hooks.transition_gate
        if page['kind'] == 'activity_inventory':
            index = int(gate.pending.target.key.split(':')[1]) if gate.pending is not None and gate.pending.target.kind == 'recovery_inventory_detail' else None
            inventory = self.inventory.observe(parent.context, images, rows, tuple(keys), inspect_index=index)
            if index is not None and inventory.get('selected_index') != index:
                raise ValueError('activity census original detail is not yet confirmed')
            navigation = self.inventory.next_navigation()
            if navigation is not None:
                page.update(target=navigation['target'], target_kind=navigation['kind'], target_key=navigation['key'])
            page['inventory'] = dict(inventory, census_id=value['transaction_id'])
        elif value is not None and page['kind'] in ('activity_home', 'consult_shop', 'support_lesson'):
            if page['kind'] == 'support_lesson' and activity_reward.observed_lesson_grants(value):
                return
            page['drink_identities'] = deepcopy(value['drink_identities'])
            if (value.get('inventory') or {}).get('complete') is not True:
                from .production import recognize_route_inventory_button as recognize_inventory_button
                recognize = support_lesson_inventory_button if page['kind'] == 'support_lesson' else recognize_inventory_button
                matches, _, _, _ = recognize(parent.context, images[-1], tuple(parent.context.gkh_hif_client_size))
                if len(matches) != 1:
                    error = ValueError('activity current owned-card control is missing' if not matches
                        else 'activity current owned-card control is ambiguous')
                    error.evidence = dict(page_kind=page['kind'], candidate_count=len(matches),
                        boxes=[list(row.box) for row in matches])
                    raise error
                page.update(target=tuple(matches[0].box), target_key='recovery_inventory', target_kind='recovery_inventory')
            else:
                held = {drink.slot for drink in host.ledger.pending.before.drinks}
                unknown = [row for row in page['drink_slots'] if row['state'] == 'OCCUPIED'
                    and row['slot'] not in held and str(row['slot']) not in value['drink_identities']]
                for key in ('target', 'target_key', 'target_kind'): page.pop(key, None)
                if unknown:
                    page.update(target=tuple(unknown[0]['box']), inspect_slot=unknown[0]['slot'],
                        target_key='activity_open_drink_detail', target_kind='activity_open_drink_detail')

    def observe(self, *, reconciliation=False, _deadline=None):
        from .calendar import ScheduleCountdownChanged, refresh_schedule_home
        from .production import _current_rows, nonbattle_drink_slots_consistent
        parent, host = self.parent, self.parent.host
        state, gate = host.ledger.state, host.hooks.transition_gate
        identity = (host.handoff or {}).get('recovery_identity', {})
        if (host.ownership['owner'] != 'task600' or host.ownership['phase'] != 'nonbattle'
                or state.day not in ((1, 4) if (activity_reward.request(host) or {}).get('kind') == 'support' else (3, 6))
                or identity.get('run_id') != host.run_id
                or (identity.get('idol'), identity.get('plan')) != active_hif_role(host)
                or identity.get('max_stamina') != state.max_stamina):
            raise ValueError('activity lacks its original bound permanent ledger and identity')
        deadline = parent.clock.now() + (gate.timeout_ms if gate.pending is not None and not reconciliation else 8000) if _deadline is None else _deadline
        if gate.pending is not None and not reconciliation: deadline = min(deadline, gate.issued_at_ms + gate.timeout_ms)
        pages, keys, images, rows, boxes = [], [], [], [], {}
        while True:
            started = parent.clock.now()
            if started >= deadline or parent.context.tasker.stopping: raise ValueError('activity original observation deadline reached')
            capture = parent.context.tasker.controller.post_screencap().wait()
            if not capture.succeeded: raise ValueError('activity capture failed')
            image = capture.get()
            key = f'route-frame-{parent.game_pid}-{started}'
            text = _current_rows(parent.context, image)
            countdown_frame_id = None
            try:
                refreshed = None
                if (not reconciliation and state.phase == 'schedule' and gate.pending is None
                        and host.ledger.pending is None and activity_reward.request(host) is None):
                    refreshed = refresh_schedule_home(parent, image, text, key, day=state.day)
                if refreshed is not None:
                    countdown_frame_id = key
                    started, image, text, calendar = refreshed
                    key = f'route-frame-{parent.game_pid}-{started}'
                    page = self._read(image, text, calendar=calendar)
                else:
                    page = self._read(image, text)
                if (not reconciliation and gate.pending is not None
                        and (gate.pending.target.kind, page['kind']) in (
                            ('confirm_outing', 'outing'), ('confirm_consult', 'consult_entry'))):
                    raise ValueError('activity confirmation is waiting for its destination')
                if parent.evidence_sink: parent.evidence_sink(image, dict(frame_id=key, phase=page['kind'], observation=page, rows=text))
                pages, keys, images, rows = [page], [key], [image], [text]
                if page['kind'] in ('card_offer', 'card_selected', 'card_inspection', 'drink_offer', 'drink_selected', 'drink_inspection'):
                    self._complete_offer(pages, images, rows)
                if 'drink_slots' in page and not nonbattle_drink_slots_consistent(page['drink_slots']):
                    raise ValueError('activity physical drink controls are unresolved')
                self._complete_census(pages, images, rows, keys)
                boxes = {page['target_key']: tuple(page['target'])} if 'target' in page else {}
                if (boxes and page['kind'] == 'support_reward'
                        and (activity_reward.request(host) or {}).get('source_target') == 'lesson:Vi'
                        and page['grant'].get('kind') != 'upgrade'):
                    boxes[page['target_key']] = ACQUIRE_SKIP
                break
            except ScheduleCountdownChanged:
                pages, keys, images, rows = [], [], [], []
                continue
            except ValueError as error:
                from .candidate_pool import CandidatePoolObserved
                from .produce_settings import ProduceSettingsObserved

                if isinstance(error, (CandidatePoolObserved, ProduceSettingsObserved)):
                    raise
                metadata = dict(frame_id=key, phase='unknown', reason=str(error), rows=text)
                if isinstance(getattr(error, 'evidence', None), dict): metadata['field_evidence'] = error.evidence
                parent.events.append(metadata)
                if parent.evidence_sink: parent.evidence_sink(image, metadata)
                if gate.pending is None: raise
                pages, keys, images, rows = [], [], [], []
                time.sleep(.1)
        value = activity_reward.request(host)
        page = pages[-1]
        kind = page['kind']
        if kind == 'delete_choice':
            from .rewards import consult_exit_request
            from .deletion_production import bind_course_deletion
            page = bind_course_deletion(parent, images, rows, tuple(keys), consult_exit_request(host, gate), page)
            revision = parent.clock.now()
            observation = Observation(host.run_id, revision, 'reward_delete_choice',
                f'consult-exit-delete:{host.ledger.state.revision}', 6, *active_hif_role(host),
                revision-started, len(keys), 'HIF')
            return HifRoutePage(observation, {}, baseline=host.ledger.state, frame_ids=tuple(keys),
                reward=dict(page, frame_ids=list(keys)))
        target_kind = page.get('target_kind', page.get('target_key'))
        revision = parent.clock.now()
        observation = Observation(host.run_id, revision, kind,
            activity_page_id(state.day, kind, (value or {}).get('transaction_id', 'calendar'), page),
            page['day'], *active_hif_role(host), revision-started, len(keys), 'HIF',
            tuple(Target(key, target_kind, available=True) for key in boxes))
        effects, receipt = (), None
        if value is None:
            from .outing import outing_effect
            from .consult import consult_effect
            if boxes and host.ledger.pending is None:
                effect = outing_effect(state, page['target_key'], dict(page, run_id=host.run_id)) if state.day == 3 else consult_effect(
                    state,page['target_key'],single_entry=True)
                effects = (effect,)
            if gate.pending is not None and host.ledger.pending is not None and kind in DESTINATIONS.get(gate.pending.target.kind, ()):
                expected = host.ledger.pending.expected
                if (page['stamina'], page['max_stamina'], page['currency'], page['day']) != (
                        expected.stamina, expected.max_stamina, expected.currency, expected.day):
                    raise ValueError('activity zero-entry destination does not prove its original resource baseline')
                receipt = expected
        proof = dict(page, frame_ids=list(keys))
        parent.events.append(dict(phase=kind, frame_ids=list(keys), capture_age_ms=observation.capture_age_ms))
        parent.last_frame_ids = tuple(keys)
        return HifRoutePage(observation, boxes, baseline=state, frame_ids=tuple(keys), effects=effects,
            receipt=receipt, receipt_frame_ids=tuple(keys) if receipt is not None else (), activity=proof,
            schedule_countdown_frame_id=countdown_frame_id if kind in ('outing', 'consult_entry') else None)

    def prepare(self, gate, report):
        if gate.pending is None or gate.pending.target.kind not in DESTINATIONS:
            raise ValueError('activity recovery has no original supported submitted input')
        deadline = self.parent.clock.now() + 8000
        proof = self.observe(reconciliation=True, _deadline=deadline)
        if (proof.reward or {}).get('kind') == 'delete_choice':
            return self.parent.host.reconcile_activity_navigation(gate, report, proof)
        value = activity_reward.request(self.parent.host)
        if (gate.pending.target.kind == 'select_consult' and proof.observation.scene == 'consult_selected'
                and value is not None and value.get('source_target') == 'select_consult'
                and value['stage'] == 'paid_pending'):
            from .daily import reconcile_unselected_preview
            return reconcile_unselected_preview(self.parent.host,gate,report,proof,target_kind='select_consult',
                unselected_scene='consult_selected',source_phase='schedule',confirmed_phase='consult_selected',
                outcome='verified_selected_preview_after_expired_single_consult_entry')
        before = {'confirm_consult':('consult_entry','consult_selected'), 'confirm_outing':('outing','outing_selected')}
        if gate.pending.target.kind in before and proof.observation.scene == before[gate.pending.target.kind][0]:
            from .daily import reconcile_unselected_preview
            scene,phase = before[gate.pending.target.kind]
            return reconcile_unselected_preview(self.parent.host,gate,report,proof,
                target_kind=gate.pending.target.kind,unselected_scene=scene,source_phase=phase,
                outcome='verified_unselected_after_free_activity_confirmation')
        while proof.activity['kind'] == 'consult_reward_transition':
            self.parent.host._page = proof
            self.parent.host.bind_observed_activity_transition(proof)
            proof = self.observe(reconciliation=True, _deadline=deadline)
        return self.parent.host.reconcile_activity_navigation(gate, report, proof)

    def decide(self, observation, config):
        return decide_activity_reward(self.parent.host, observation, config)
