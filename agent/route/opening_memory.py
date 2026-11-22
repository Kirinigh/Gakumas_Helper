"""Observe opening memory effects without making them a route decision."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .engine import has_single_frame_id


def read_memory_effect(rows, catalog):
    """Use the acquire title and the visible memory-effect heading from Hiro."""
    from .activity_production import canonical

    def in_band(row, low, high):
        return row['score'] >= .95 and low <= row['box'][1] and row['box'][1] + row['box'][3] <= high

    if not any(in_band(row, 400, 500) and 120 <= row['box'][0]
            and row['box'][0] + row['box'][2] <= 620
            and canonical(row['text']) == 'メモリー効果' for row in rows):
        return None
    titles = {canonical(row['text']) for row in rows if in_band(row, 840, 910)
        and 345 <= row['box'][0] + row['box'][2] / 2 <= 395}
    entries = [entry for entry in catalog.entries.values() if canonical(entry.name) in titles]
    if len(entries) != 1:
        return dict(card_id=None, before_card_id=None, kind=None)
    card = entries[0]
    family = canonical(card.name).removesuffix('+')
    base = [entry for entry in catalog.entries.values() if not entry.upgraded and entry.plan == card.plan
        and canonical(entry.name) == family]
    base_id = base[0].card_id if card.upgraded and len(base) == 1 else None
    words = set()
    for row in rows:
        if not in_band(row, 720, 1110):
            continue
        text = re.fullmatch(r'(.+?)を(獲得|強化)しました[。]?', canonical(row['text']))
        if text is not None and text[1] in (canonical(card.name), family):
            words.add(text[2])
    kind = None
    if words == {'獲得'} or not words and not card.upgraded:
        kind = 'add'
    elif words == {'強化'} and base_id is not None:
        kind = 'upgrade'
    return dict(card_id=card.card_id, before_card_id=base_id, kind=kind)


def _source_scope(start):
    from .new_run import _memory_stats

    request = start.get('request') or {}
    memory = start.get('selected_memory') or {}
    frames = memory.get('frame_ids') or ()
    run_id = start.get('run_id')
    pid = start.get('game_pid')
    issued = request.get('issued_at_ms')
    if (isinstance(run_id, str) and run_id and type(pid) is int and type(issued) is int
            and has_single_frame_id(frames) and run_id == memory.get('run_id')
            and _memory_stats(memory.get('stats')) is not None
            and start.get('status') in ('reserved', 'created_ap_unverified', 'complete')):
        return (run_id, pid, issued)
    return None


def _current_frame(frame, scope):
    if not isinstance(frame, str) or not frame.startswith(f'route-frame-{scope[1]}-'):
        return False
    tick = frame.removeprefix(f'route-frame-{scope[1]}-')
    return tick.isdecimal() and int(tick) >= scope[2]


def observe_opening_memory(producer, rows):
    host = producer.host
    if (host is None or not getattr(host, '_active', False) or host.ledger is not None
            or host._catalog is None or not any('メモリー効果' in row['text'] for row in rows)):
        return
    start = (host.handoff or {}).get('cultivation_start') or {}
    scope = _source_scope(start)
    frame = producer.current_frame_id
    if scope is None or scope[:2] != (producer.run_id, producer.game_pid) or not _current_frame(frame, scope):
        return
    read = read_memory_effect(rows, host._catalog)
    if read is None:
        return
    receipts = _scoped_receipts(start, scope)
    # Frame repeats only corroborate the displayed variant; they cannot count
    # multiple identical awards or establish their order.
    if not any(row['card_id'] == read['card_id'] and row['kind'] == read['kind'] for row in receipts):
        start['memory_effect_receipts'] = [*receipts, dict(read, frame_ids=[frame])]


def _scoped_receipts(start, scope):
    receipts = start.get('memory_effect_receipts')
    if not isinstance(receipts, list):
        return []
    return [row for row in receipts if isinstance(row, dict)
        and {'card_id', 'before_card_id', 'kind', 'frame_ids'} <= row.keys()
        and row['kind'] in (None, 'add', 'upgrade') and has_single_frame_id(row['frame_ids'])
        and _current_frame(row['frame_ids'][0], scope)]


def opening_memory_receipts(start):
    scope = _source_scope(start)
    return _scoped_receipts(start, scope) if scope is not None and start['status'] == 'complete' else []


def reconcile_opening_memory(receipts, cards, *, before=None):
    """Bind result cards by instance. An untyped receipt needs the prior deck."""
    grouped = defaultdict(list)
    for receipt in receipts:
        grouped[receipt['card_id']].append(receipt)
    original = Counter(row['card_id'] for row in before) if before is not None else None
    result = []
    for card_id, evidence in grouped.items():
        variant = next((row['before_card_id'] for row in evidence if row['before_card_id'] is not None), None)
        matches = [row for row in cards if card_id is not None and row['card_id'] == card_id]
        stated = {row['kind'] for row in evidence if row['kind'] is not None}
        kind = next(iter(stated)) if len(stated) == 1 else None
        if original is not None and len(stated) < 2 and card_id is not None:
            lower = sum(row['card_id'] == variant for row in cards) - original[variant] if variant is not None else 0
            higher = len(matches) - original[card_id]
            inferred = 'upgrade' if (lower, higher) == (-1, 1) else 'add' if (lower, higher) == (0, 1) else None
            kind = inferred if kind in (None, inferred) else None
        if len(matches) != 1:
            kind = None
        result.append(dict(source='memory', timing='start', kind=kind or 'unknown', card_id=card_id,
            before_card_id=variant if kind == 'upgrade' else None,
            card_ref=matches[0]['instance_id'] if kind else None,
            frame_ids=[key for row in evidence for key in row['frame_ids']]))
    return result
