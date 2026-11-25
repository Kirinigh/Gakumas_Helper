"""Export a redacted HIF route and permanent-card replay from local checkpoints."""
from __future__ import annotations

import re
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'agent'))

from route.deck_history import record_exam_deck, read_deck_history, record_deck_settlement  # noqa: E402


def _receipt(handoff, transaction_id):
    while isinstance(handoff, dict):
        handoff = handoff.get('nonbattle_context')
        if not isinstance(handoff, dict):
            break
        for key in ('activity_reward', 'training_reward'):
            receipt = handoff.get(key) or {}
            if (receipt.get('stage') == 'complete' and receipt.get('transaction_id') == transaction_id):
                return receipt
    return {}


def _archive_source(transaction_id, run_id, effect):
    if effect:
        return effect['target_key']
    # The original ledger embeds a route action in its transaction key. Only
    # this canonical segment survives export, followed by a separate alias.
    suffix = transaction_id.removeprefix(run_id + ':').split(':')
    if len(suffix) >= 2 and suffix[0].isdigit():
        name = suffix[1].replace('-', '_')
        return ':'.join([name, *suffix[2:3]]) if name in ('lesson', 'select_lesson', 'confirm_lesson') else name
    return None


def replay_archive(records):
    """Reconstruct the supported chronology without treating a snapshot as an action."""
    run_id = history = previous = None
    pending_effects = {}
    for record in records:
        if run_id is None:
            run_id = record['run_id']
        if record['run_id'] != run_id:
            raise ValueError('archive includes more than one cultivation')
        ledger = record['ledger']
        if ledger is None:
            continue
        state, settled = ledger['state'], ledger['settled']
        handoff = record.get('handoff') or {}
        if previous is None:
            opening = (state['day'], state['phase']) == (1, 'schedule') and not settled
            source = {'cultivation_start': handoff.get('cultivation_start')}
            history = record_deck_settlement(source, None, state, opening=opening,
                frames=handoff.get('recovery_frame_ids', ()))
        else:
            old, old_settled = previous['state'], previous['settled']
            if settled[:len(old_settled)] != old_settled or state['revision'] < old['revision']:
                raise ValueError('archive checkpoints must be supplied in their original settlement order')
            added = settled[len(old_settled):]
            changed = any(state.get(key) != old.get(key) for key in ('cards', 'card_count', 'cards_complete'))
            sources = {tx: _archive_source(tx, run_id, pending_effects.get(tx)) for tx in added}
            receipts = {tx: _receipt(handoff, tx) for tx in added}
            relevant = [tx for tx in added if (
                sources[tx] == 'execute_lesson_exchange' or any(pending_effects.get(tx, {}).get(key) for key in (
                    'add_cards', 'remove_cards', 'observed_cards')) or any(receipts[tx].get(key) for key in (
                    'cards', 'upgrades', 'deletion')))] if changed else []
            attributed = relevant[0] if len(relevant) == 1 else None
            for tx in added:
                receipt = receipts[tx]
                effect = pending_effects.get(tx, {})
                grants = receipt.get('cards') or receipt.get('upgrades') or ()
                frames = (receipt.get('resolution') or {}).get('frame_ids', ())
                before, after = old, state if tx == attributed or (len(added) == 1 and not changed) else old
                if len(added) > 1:
                    day = receipt.get('day', old['day'] if old['day'] == state['day'] else None)
                    before = dict(before, day=day, phase=None)
                    after = dict(after, day=state['day'] if tx == attributed else day, phase=None)
                history = record_deck_settlement(history, before, after,
                    source=sources[tx], transaction_id=tx, frames=frames, grants=grants,
                    observed_ids={row['instance_id'] for row in effect.get('observed_cards', ())})
            if changed and attributed is None:
                history = record_deck_settlement(history, old, state, source='retained_census', census=True)
        if ledger.get('pending'):
            effect = ledger['pending']['effect']
            pending_effects[effect['transaction_id']] = effect
        if (record.get('ownership', {}).get('owner') == 'task275'
                and handoff.get('ledger_revision') == state['revision']
                and handoff.get('round_number') in (1, 2) and ledger['pending'] is None):
            history = record_exam_deck(history, state, round_number=handoff['round_number'],
                frame_ids=handoff['source_frame_ids'])
        previous = ledger
    return read_deck_history(history)


def export_history(history, *, retained=False):
    """Emit an allowlisted payload with report-local card, transaction and frame IDs."""
    if history is None:
        raise ValueError('no HIF deck history in the supplied checkpoint')
    cards, transactions, frames = {}, {}, {}

    def alias(table, value, prefix):
        return table.setdefault(value, f'{prefix}-{len(table) + 1}') if value is not None else None

    def card(row):
        if row is None:
            return None
        return dict(instance_id=alias(cards, row['instance_id'], 'card'), card_id=str(int(row['card_id'])),
            upgraded=row['upgraded'], customization=[[str(int(k)), int(n)] for k, n in row['customization']])

    def snapshot(row):
        return dict(cards=[card(item) for item in row['cards']], total=row['total'], complete=row['complete'],
            deck_order='unknown', evidence=[alias(frames, key, 'frame') for key in row['frame_ids']])

    baseline = dict(snapshot(history['baseline']), day=history['baseline']['day'], origin=history['origin'])
    opening = history['baseline'].get('opening_sources')
    if opening is not None:
        effects = opening.get('memory_effects')
        baseline['opening_sources'] = dict(coverage='partial', starting_cards=None,
            qualifier_confirmed=opening['qualifier_memory_stats'] is not None,
            memory_effect_order='unknown', memory_effects=None if effects is None else [dict(
                source='memory', timing='start', kind=row['kind'],
                card_id=str(int(row['card_id'])) if row['card_id'] is not None else None,
                before_card_id=str(int(row['before_card_id'])) if row['before_card_id'] is not None else None,
                card_ref=alias(cards, row['card_ref'], 'card'),
                evidence=[alias(frames, key, 'frame') for key in row['frame_ids']]) for row in effects])
    events, exams = [], []
    current = {row['instance_id']: row for row in baseline['cards']}
    positions = {0: (current.copy(), baseline['total'], baseline['complete'])}
    for row in history['events']:
        source = row.get('source') or ''
        if not re.fullmatch(r'[a-z][a-z_0-9-]*(?::(?:Vo|Da|Vi))?', source):
            source = None
        changes = [dict(kind=change['kind'], before=card(change['before']), after=card(change['after']))
            for change in row['changes']]
        event = {key: row[key] for key in ('sequence', 'day', 'phase', 'next_day', 'next_phase',
            'kind', 'order', 'total', 'complete')}
        events.append(dict(event, source=source, transaction=alias(transactions, row['transaction_id'], 'tx'),
            evidence=[alias(frames, key, 'frame') for key in row['frame_ids']], changes=changes))
        for change in changes:
            first, last = change['before'], change['after']
            key = (first or last)['instance_id']
            if current.get(key) != first:
                raise ValueError(f'deck replay before-state differs at event {row["sequence"]}')
            if first is not None:
                current.pop(key)
            if last is not None:
                current[key] = last
        positions[row['sequence']] = (current.copy(), row['total'], row['complete'])
    for row in history['exams']:
        deck = snapshot(row)
        through = row['through_sequence']
        replay, total, complete = positions[through]
        exam = {item['instance_id']: item for item in deck['cards']}
        match = replay == exam and (total is None or deck['total'] is None or total == deck['total'])
        status = 'mismatch' if not match else 'partial'
        if match and complete and deck['complete'] and len(exam) == total == deck['total']:
            status = 'match'
        exams.append(dict(deck, round_number=row['round_number'], through_sequence=through, reconciliation=status))
    return dict(schema_version=1, run='hif-replay-1', provenance='archived_checkpoints' if retained else 'settled_hif_record',
        baseline=baseline, events=events, exams=exams)


def _hiro_card(args, id_key='card_id', upgrade_key='upgrade'):
    name, upgrade = args.get(id_key), args.get(upgrade_key)
    if not isinstance(name, str) or not re.fullmatch(r'p_card-[a-zA-Z0-9_/-]+', name):
        return None
    return dict(card_id=name, upgrade_count=upgrade if type(upgrade) is int and upgrade >= 0 else None)


_HIRO_CARD_CHANGES = {'CARD_ACQUIRE', 'CARD_DELETE', 'CARD_CHANGE', 'CARD_UPGRADE',
                     'CARD_CUSTOMIZE', 'CARD_AUTO_PROCESS', 'UNMODELED_CHANGE', 'SHOP_BUY'}


def _hiro_delta(event, page):
    kind, args = event['kind'], event['args']
    card = _hiro_card(args)
    if card is None:
        return None
    if kind in ('CARD_ACQUIRE', 'SHOP_BUY'):
        if args.get('memory_effect'):
            before = event.get('memory_upgrade_from')
            if type(before) is int and type(card['upgrade_count']) is int and 0 <= before < card['upgrade_count']:
                return dict(op='modify', before=dict(card, upgrade_count=before), after=card)
            return None
        context = 'shop' if kind == 'SHOP_BUY' else 'reward' if page == 'SELECT_CARD' else 'other'
        return dict(op='add', after=card, context=context)
    if kind == 'CARD_DELETE':
        return dict(op='remove', before=card, context='shop' if args.get('source') == 'consult' else 'other')
    if kind == 'CARD_CHANGE':
        before = _hiro_card(args, 'remove_card_id', 'remove_upgrade')
        return dict(op='replace', before=before, after=card, context='lesson') if before else None
    if kind == 'CARD_UPGRADE' and card['upgrade_count'] is not None:
        level = card['upgrade_count']
        if event['origin'] == 'action':
            return dict(op='modify', before=card, after=dict(card, upgrade_count=level + 1))
        if level > 0:
            return dict(op='modify', before=dict(card, upgrade_count=level - 1), after=card)
    return None


def export_hiro_history(report, *, segment):
    """Project one deck lifetime to the exam-observation v1 exchange format."""
    if segment not in report['starts']:
        raise ValueError('the requested HIF segment has no opening receipt')
    events, attempts = [], {}
    applied = set()
    last_delta = 0
    census = None
    replay = None

    def write(kind, source='screen', **data):
        events.append(dict(seq=len(events) + 1, type=kind, source=source, **data))

    def gap(reason):
        nonlocal census, replay
        census = replay = None
        write('gap', scope='deck', reason=reason)

    def bag(cards):
        return Counter((item['card_id'], item['upgrade_count']) for item in cards)

    def step(row):
        nonlocal last_delta, census, replay
        changes, reviews = [], []
        for event in row['events']:
            kind, args = event['kind'], event['args']
            if kind == 'DECK_REVIEW' and event['origin'] == 'review' and event['confirmed'] is True:
                cards = [_hiro_card(item) for item in args['deck']]
                if args.get('deck_size') == len(cards) and all(cards):
                    reviews.append(cards)
                continue
            if (kind not in _HIRO_CARD_CHANGES
                    or kind == 'SHOP_BUY' and (args.get('kind') == 'drink' or args.get('drink_id'))
                    or kind == 'UNMODELED_CHANGE' and 'deck' not in str(args.get('field', 'deck'))):
                continue
            paired = (event.get('ref') or {}).get('paired')
            if (paired, args.get('card_id')) in applied:
                continue
            changes.append(event)
        if len(changes) > 1 or changes and reviews:
            gap('unknown_order')
        elif reviews:
            observed = bag(reviews[-1])
            if replay is not None and replay != observed:
                gap('unknown_order')
            census, replay = reviews[-1], observed
        elif changes:
            event, = changes
            execution = (event['origin'] == 'action' and row['sequence'] == event['seq']
                         and (row.get('execution') or {}).get('ok') is True)
            if event.get('confirmed') is not True and not execution:
                gap('unconfirmed_result')
                return
            delta = _hiro_delta(event, row['page'])
            if delta is None:
                gap('other' if event['kind'] == 'CARD_CUSTOMIZE' else 'unknown_identity')
                return
            if replay is not None:
                before = bag([delta['before']]) if 'before' in delta else Counter()
                after = bag([delta['after']]) if 'after' in delta else Counter()
                if before - replay:
                    gap('unknown_identity')
                    return
                replay = replay - before + after
            write('deck_delta', 'verified_execution' if execution else 'screen', **delta)
            applied.add((event['seq'], event['args'].get('card_id')))
            last_delta = len(events)
            census = None

    rows = [row for row in report['events'] if row['segment'] == segment]
    index = 0
    write('gap', scope='deck', reason='missing_observation')
    for exam in report['exams']:
        kind, label = exam['event'], exam.get('label')
        if kind == 'hiro_exam_deck' and exam.get('segment') == segment:
            through = exam['through_event']
            if not index <= through <= len(rows):
                raise ValueError('HIF exam cutoffs are out of recording order')
            for row in rows[index:through]:
                step(row)
            index = through
            previous = attempts.get(label)
            attempt = attempts[label] = f'{label}-{len(events) + 1}'
            known = [_hiro_card(item) for item in exam.get('cards') or ()]
            data = dict(attempt_id=attempt, stage=label, deck_history='partial', deck_through_seq=last_delta)
            if census:
                data['deck_check'] = census
            if replay is not None and known and all(known) and bag(known) != replay:
                gap('unknown_order')
            if previous:
                data['retry_of_attempt_id'] = previous
            write('exam_start', **data)
        elif kind == 'cultivation_exam_result':
            if label in attempts:
                write('exam_end', attempt_id=attempts[label], outcome='completed', score=exam.get('score'))
        elif kind == 'hiro_formal_hif_result' and exam['status'] == 'TERMINAL_VERIFIED':
            label = f'r{exam["round_number"]}'
            if label in attempts:
                write('exam_end', attempt_id=attempts[label], outcome='completed', score=exam.get('score'))
    for row in rows[index:]:
        step(row)
    return dict(format='gakumas.exam-observation', schema_version=1, run_id=f'hif-replay-{segment}',
        master_version=None, capture_start='run_start',
        loadout=dict(mode='hif', difficulty_id=None, idol_id=report.get('idol'), starting_cards=None),
        events=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, nargs='+', help='one current checkpoint or archived checkpoints in original order')
    parser.add_argument('--archive', action='store_true', help='reconstruct from historical checkpoint sequence')
    parser.add_argument('--hiro-segment', choices=('qualify', 'main'), help='export one Hiro deck lifetime in exchange format')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.hiro_segment and args.archive:
        parser.error('--hiro-segment requires a single Hiro report')
    if not args.archive and len(args.checkpoint) != 1:
        parser.error('one checkpoint required without --archive')
    records = (json.loads(path.read_bytes()) for path in args.checkpoint)
    if args.hiro_segment:
        output = export_hiro_history(next(records), segment=args.hiro_segment)
    else:
        history = replay_archive(records) if args.archive else read_deck_history(next(records)['handoff'])
        output = export_history(history, retained=args.archive)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
