"""Permanent-card changes in the existing HIF settlement and handoff record."""
from __future__ import annotations

from dataclasses import asdict

KEY = 'deck_history'


def read_deck_history(handoff):
    while isinstance(handoff, dict):
        context = handoff.get('nonbattle_context')
        if not isinstance(context, dict):
            break
        value = context.get(KEY)
        if (isinstance(value, dict) and value.get('schema_version') == 1
                and isinstance(value.get('baseline'), dict) and isinstance(value.get('events'), list)
                and isinstance(value.get('exams'), list)):
            return value
        handoff = context
    return None


def _store(handoff, history):
    root = dict(handoff or {})
    root['nonbattle_context'] = dict(root.get('nonbattle_context') or {}, **{KEY: history})
    return root


def _snapshot(state):
    return dict(cards=[dict(row, customization=[list(pair) for pair in row.get('customization', ())])
            for row in sorted(state['cards'], key=lambda card: card['instance_id'])],
        total=state['card_count'], complete=state.get('cards_complete', True))


def _baseline(state, *, opening=False, frames=()):
    return dict(schema_version=1, origin='opening' if opening else 'first_retained_state',
        baseline=dict(_snapshot(state), day=state['day'], revision=state['revision'], frame_ids=list(frames)),
        events=[], exams=[])


def _opening_sources(handoff, state):
    from .engine import has_single_frame_id
    from .new_run import _memory_stats
    from .opening_memory import opening_memory_receipts, reconcile_opening_memory

    run_id = state.get('run_id')
    start = (handoff or {}).get('cultivation_start') or {}
    memory = start.get('selected_memory') or {}
    frames = memory.get('frame_ids') or ()
    stats = _memory_stats(memory.get('stats'))
    if (run_id is None or start.get('run_id') != run_id or memory.get('run_id') != run_id
            or not has_single_frame_id(frames) or stats is None):
        stats, frames = None, ()
    receipts = opening_memory_receipts(start) if stats is not None else []
    cards = state['cards'] if state.get('cards_complete', True) else ()
    return dict(qualifier_memory_stats=list(stats) if stats is not None else None,
        qualifier_memory_frames=list(frames), starting_cards=None, coverage='partial',
        memory_effects=reconcile_opening_memory(receipts, cards) if receipts else None)


def _diff(before, after, observed_ids, *, census):
    old = {card['instance_id']: card for card in before['cards']}
    new = {card['instance_id']: card for card in after['cards']}
    result = []
    for uid in sorted(old.keys() | new.keys()):
        first, last = old.get(uid), new.get(uid)
        if first == last:
            continue
        if uid in observed_ids or (census and first is None):
            kind = 'observed'
        elif census:
            kind = 'unknown'
        elif first is None:
            kind = 'acquire'
        elif last is None:
            kind = 'delete'
        elif (first['card_id'], first['upgraded']) != (last['card_id'], last['upgraded']):
            kind = 'upgrade' if not first['upgraded'] and last['upgraded'] else 'unknown'
        else:
            kind = 'customize'
        result.append(dict(kind=kind, before=first, after=last))
    return result


def _ordered_grants(changes, grants):
    """Associate an ordered list of confirmed grants with unique instances."""
    if not grants or len(grants) != len(changes):
        return None
    remaining = list(changes)
    ordered = []
    for grant in grants:
        upgrade = grant.get('kind') == 'upgrade'
        matches = [row for row in remaining if row['kind'] == ('upgrade' if upgrade else 'acquire')
            and row['after']['card_id'] == grant.get('card_id')
            and (not upgrade or (row['before']['card_id'] == grant.get('source_card_id')
                and grant.get('source_instance_id') in (None, row['before']['instance_id'])))]
        if len(matches) != 1:
            return None
        ordered.append(matches[0])
        remaining.remove(matches[0])
    return ordered


def record_deck_settlement(handoff, before, after, *, source=None, transaction_id=None,
        frames=(), grants=(), observed_ids=(), opening=False, census=False):
    """Add evidence to the same candidate as the settled ledger. No UI input is needed."""
    history = read_deck_history(handoff)
    if before is None:
        if history is not None:
            return handoff
        history = _baseline(after, opening=opening, frames=frames)
        if opening:
            history['baseline']['opening_sources'] = _opening_sources(handoff, after)
        return _store(handoff, history)
    old, new = _snapshot(before), _snapshot(after)
    if old == new and transaction_id is None:
        return handoff
    history = history or _baseline(before)
    if transaction_id is not None and any(row.get('transaction_id') == transaction_id for row in history['events']):
        return handoff
    changes = _diff(old, new, observed_ids, census=census)
    ordered = _ordered_grants(changes, grants)
    exchange = (source == 'execute_lesson_exchange'
        and sorted(row['kind'] for row in changes) == ['acquire', 'delete'])
    order = ('unknown' if census else 'simultaneous' if exchange else
        'confirmed' if ordered is not None or len(changes) <= 1 else 'unknown')
    event = dict(sequence=len(history['events']) + 1, day=before['day'], phase=before['phase'],
        next_day=after['day'], next_phase=after['phase'],
        before_revision=before['revision'], revision=after['revision'], source=source,
        transaction_id=transaction_id, frame_ids=list(frames),
        kind='census' if census else 'exchange' if exchange else 'route' if old == new else 'settlement',
        order=order, changes=ordered if ordered is not None and not census else changes,
        total=new['total'], complete=new['complete'])
    return _store(handoff, dict(history, events=[*history['events'], event]))


def record_ledger_deck(handoff, before, after, *, frames=(), grants=(), opening=False, census=False):
    """Read already confirmed state and the original pending effect, if any."""
    prior = before.state if before is not None else None
    state = after.state
    effect = before.pending.effect if before is not None and before.pending is not None else None
    if not census and (effect is None or effect.transaction_id not in after.settled):
        census = True
    if (census and prior is not None and prior.cards == state.cards and prior.card_count == state.card_count
            and getattr(prior, 'cards_complete', True) == getattr(state, 'cards_complete', True)):
        return handoff
    observed = {row.instance_id for row in getattr(effect, 'observed_cards', ())}
    observed.update(row.instance_id for row in state.cards if row.instance_id.startswith(f'{state.run_id}:census:'))
    return record_deck_settlement(handoff, asdict(prior) if prior is not None else None, asdict(state),
        source=effect.target_key if effect is not None else 'deck_census',
        transaction_id=effect.transaction_id if effect is not None and not census else None,
        frames=frames, grants=grants, observed_ids=observed, opening=opening, census=census)


def record_exam_deck(handoff, state, *, round_number, frame_ids):
    """Attach the cutoff to the same publication as the existing battle handoff."""
    history = read_deck_history(handoff) or _baseline(state)
    if any(row.get('round_number') == round_number for row in history['exams']):
        return handoff
    exam = dict(_snapshot(state), round_number=round_number, revision=state['revision'],
        through_sequence=len(history['events']), frame_ids=list(frame_ids))
    return _store(handoff, dict(history, exams=[*history['exams'], exam]))
