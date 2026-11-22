"""Bind a fresh post-battle permanent census to this exact recovery epoch."""
from __future__ import annotations

from copy import deepcopy
from collections import defaultdict

from .ledger import CardInstance, PartialDeckLedgerState


def scope(host):
    owner, handoff = host.ownership or {}, host.handoff or {}
    if (owner.get('owner'), owner.get('phase'), owner.get('game_pid')) != ('task600', 'recovery', host.game_pid):
        raise ValueError('nonbattle census lacks its actual current recovery owner')
    terminal = handoff.get('terminal_receipt') or {}
    return dict(schema_version=1, run_id=host.run_id, game_pid=host.game_pid,
        ownership_revision=owner['revision'], prior_ledger_revision=host.ledger.state.revision if host.ledger else None,
        terminal_source=handoff.get('terminal_source'), round_number=terminal.get('round_number'),
        live_run_id=terminal.get('run_id'), live_revision=terminal.get('revision'))


def retained(host, *, complete=False):
    value = (host.handoff or {}).get('nonbattle_inventory')
    if (not isinstance(value, dict) or value.get('recovery_scope') != scope(host)
            or complete and value.get('complete') is not True):
        return None
    return deepcopy(value)


def current_record(host, inventory):
    value = deepcopy(inventory)
    value['recovery_scope'] = scope(host)
    return value


def retained_identity(host):
    """Retain an actually confirmed notebook only within its unchanged epoch."""
    from .hif import known_hif_role
    handoff = host.handoff or {}
    identity, frames = handoff.get('recovery_identity'), handoff.get('identity_frame_ids')
    if handoff.get('recovery_identity_scope') != scope(host): return None
    if (not isinstance(identity, dict) or identity.get('run_id') != host.run_id
            or not known_hif_role(identity.get('idol'), identity.get('plan'))
            or type(identity.get('max_stamina')) is not int or not 1 <= identity['max_stamina'] <= 100
            or not isinstance(frames, list) or len(frames) != 1 or not isinstance(frames[0], str) or not frames[0].strip()):
        raise ValueError('retained recovery identity lacks its exact confirmed current notebook proof')
    return deepcopy(identity), tuple(frames)


def _card_variant(host, row):
    entry = host._catalog.entries.get(row.get('card_id'))
    customs = row.get('customizations')
    if (entry is None or not isinstance(customs, dict)
            or row.get('presence') not in ('CONFIDENT_ZERO', 'DETAIL_CANDIDATE')
            or row['presence'] == 'DETAIL_CANDIDATE' and (row.get('detail') or {}).get('nonbattle_schema') != 1):
        return None
    return row['card_id'], entry.upgraded, tuple(sorted((str(k), count) for k, count in customs.items()))


def partial_permanent_cards(host, inventory):
    """A single first-day read only supplies the individually resolved slots."""
    return tuple(CardInstance(f'{host.run_id}:card:{row["index"]}', *variant)
        for row in inventory['slots'] if not row.get('identity_needs_detail')
        and (variant := _card_variant(host, row)) is not None)


def permanent_cards(host, inventory):
    if (inventory.get('complete') is not True or type(inventory.get('total')) is not int
            or inventory['total'] != len(inventory.get('slots', ()))
            or [row.get('index') for row in inventory['slots']] != list(range(inventory['total']))):
        raise ValueError('post-battle recovery requires a complete current permanent census')
    old = host.ledger
    if old is not None and old.pending is not None:
        raise ValueError('recovery cannot discard a pending permanent-card transaction')
    groups = defaultdict(list)
    if old is not None:
        for card in old.state.cards: groups[card.card_id, card.upgraded, card.customization].append(card)
    partial = old is not None and isinstance(old.state, PartialDeckLedgerState)
    if partial and old.state.card_count is not None and inventory['total'] != old.state.card_count:
        raise ValueError('permanent census disagrees with the observed card total')
    result = []
    for row in inventory['slots']:
        key = _card_variant(host, row)
        if key is None:
            raise ValueError('recovery permanent card variant is unresolved')
        if old is None:
            card = CardInstance(f'{host.run_id}:card:{row["index"]}', *key)
        elif groups[key]:
            card = groups[key].pop(0)
        elif partial:
            card = CardInstance(f'{host.run_id}:census:{old.state.revision}:card:{row["index"]}', *key)
        else:
            raise ValueError('post-battle census contains an unquoted permanent addition or customization')
        result.append(card)
    if any(groups.values()): raise ValueError('post-battle census lost a permanent card or duplicate instance')
    return tuple(result)
