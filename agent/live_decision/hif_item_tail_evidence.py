"""Reuse the original 0028 item pages and the frozen reread of its failed page."""
import json
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict

import cv2

try:
    from card_selection.model import frame_identifier
except ModuleNotFoundError:
    from agent.card_selection.model import frame_identifier

from . import item_panel

SOURCE = '0028-hif-caa48d5077665badbe96d6acdb5f9295-battle-resume-live'
ARCHIVE = Path(__file__).resolve().parents[2] / '.local/task275-hif/joint-20261119'
RECEIPT = ARCHIVE / '0028-item-prefix.json'


def _source_pages(source):
    if source.get('id') != SOURCE:
        raise ValueError('item prefix belongs to another opening')
    inspection, = (s for e in source['live_events'] if e['event'] == 'live_status_observe'
                   for s in e['status_events'] if s['phase'] == 'item_description_inspection')
    if (inspection['accepted'] is not False or inspection['complete_list'] is not False
            or inspection['returned_unchanged'] is not True or len(inspection['pages']) != 3):
        raise ValueError('0028 partial item pages changed')
    return inspection['pages']


def _failure_frames(source):
    rows = [(i, row) for i, row in enumerate(source['frames']) if row['label'] == 'item-failure']
    if len(rows) != 3 or len({row['frame_id'] for _, row in rows}) != 3:
        raise ValueError('0028 item failure needs its three separate frames')
    images = []
    for i, row in rows:
        path = ARCHIVE / '0028-original' / f'{i:03}.png'
        if not path.is_file():
            path = Path(row['path'])
        image = cv2.imread(str(path)) if path.is_file() else None
        if image is None or frame_identifier(image) != row['frame_id']:
            raise ValueError('0028 retained item pixels differ from the original frame ID')
        images.append(image)
    return images


def item_prefix_consensus(record):
    """Read the observations retained with each page of the frozen prefix."""
    frames = record['frame_ids']
    if (not isinstance(frames, (list, tuple)) or not frames
            or any(not isinstance(value, str) or not value for value in frames)
            or len(set(frames)) != len(frames) or len(record['observations']) != len(frames)):
        raise ValueError('0028 item page frame identifiers differ')

    def row(raw, row_type):
        counter = item_panel.ItemRemainingCounter(**raw['remaining'])
        return row_type(**dict(raw, box=tuple(raw['box']), remaining=counter))

    observations = []
    for raw in record['observations']:
        rows = tuple(row(r, item_panel.ItemDescriptionRow) for r in raw['rows'])
        headers = tuple(row(r, item_panel.ItemDescriptionHeader) for r in raw['headers'])
        if any(not .95 <= r.confidence <= 1 or not 0 <= r.remaining.value <= r.use_limit for r in rows):
            raise ValueError('0028 counted item row has an invalid reading')
        observations.append(item_panel.ItemDescriptionPage(**dict(
            raw, thumb=tuple(raw['thumb']), rows=rows, headers=headers)))
    page = item_panel.item_panel_consensus(observations)
    for field in ('rows', 'headers'):
        for group in zip(*(getattr(o, field) for o in observations), strict=True):
            item_panel.item_counter_consensus(r.remaining for r in group)
    return page


def _verified_pages(source, receipt):
    original = _source_pages(source)
    pages = receipt['pages']
    failure_ids = [row['frame_id'] for row in source['frames'] if row['label'] == 'item-failure']
    if (receipt['source_record'] != SOURCE or receipt['handoff_id'] != source['handoff_id']
            or len(pages) != 4 or pages[:3] != original or pages[-1]['frame_ids'] != failure_ids):
        raise ValueError('0028 frozen item pages differ from their source')
    observed = [item_prefix_consensus(record) for record in pages]
    ticket, = observed[-1].rows
    if (any(p.at_bottom for p in observed)
            or (ticket.name, ticket.remaining.value) != observed[-2].headers[-1].content_key):
        raise ValueError('0028 reread does not complete the preceding partial item row')
    item_panel.join_item_description_pages(observed, complete=False)
    return pages


def freeze_item_tail_prefix(source, context):
    """Run the current item reader on original saved pixels, without an input controller."""
    original = _source_pages(source)
    frames = _failure_frames(source)
    item_panel.item_description_stable_geometry(frames)
    observations = []
    targets = []
    for image in frames:
        text = item_panel._inventory_ocr(context, image, 'LiveStatusItemDescription')
        targets.append(item_panel._item_description_target(context, image, text_rows=text))
        observations.append(asdict(item_panel.read_item_description(context, image, text_rows=text)))
    if any(target != targets[0] for target in targets[1:]):
        raise ValueError('0028 reread has no stable item panel')
    page = dict(frame_ids=[frame_identifier(image) for image in frames], observations=observations)
    receipt = dict(source_record=SOURCE, handoff_id=source['handoff_id'], pages=[*deepcopy(original), page])
    _verified_pages(source, receipt)
    return receipt


def verified_item_tail_prefix(source):
    """Load this game's frozen OCR receipt without a second recognition run or input."""
    _source_pages(source)
    try:
        receipt = json.loads(RECEIPT.read_bytes())
    except (OSError, ValueError) as error:
        raise ValueError('0028 frozen item prefix is unavailable') from error
    pages = _verified_pages(source, receipt)
    _failure_frames(source)
    return pages


def verified_closed_item_prefix(source):
    """Keep the three joined items and the separately observed fourth item."""
    inspection, = (s for e in source['live_events'] if e.get('event') == 'live_status_observe'
                   for s in e['status_events'] if s.get('phase') == 'item_description_inspection')
    pages = inspection['pages']
    if (inspection.get('returned_unchanged') is not True or inspection.get('complete_list') is not False
            or inspection.get('read_failure') != 'status list overlap is missing or ambiguous'
            or len(pages) < 2):
        raise ValueError('item inspection does not contain a closed partial list')
    observed = [item_prefix_consensus(record) for record in pages]
    previous = item_panel.join_item_description_pages(observed[:-1], complete=False)
    fourth = observed[-1]
    if (len(previous) != 3 or any(page.at_bottom for page in observed)
            or fourth.thumb[0] <= observed[-2].thumb[0] or len(fourth.rows) != 1
            or fourth.rows[0].name in {row.name for row in previous}
            or fourth.headers[0].content_key != fourth.rows[0].content_key[:2]):
        raise ValueError('item inspection does not reach a distinct fourth item')
    return dict(source_result=source['id'], slot=4, pages=deepcopy(pages[:-1]),
                observed_fourth=deepcopy(pages[-1]))
