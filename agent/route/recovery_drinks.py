"""Fresh post-battle drink names via owned, cancel-only physical-slot visits."""
from __future__ import annotations

import re
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import replace

from .ledger import DrinkInstance
from .recovery_inventory import scope

KINDS = frozenset({'recovery_drink_detail', 'recovery_drink_close'})


def request(host, *, handoff=None):
    saved = (host.handoff if handoff is None else handoff) or {}
    inventory = saved.get('nonbattle_inventory') or {}
    value = inventory.get('drink_inventory')
    if value is None: return None
    if (not isinstance(value, dict) or inventory.get('recovery_scope') != scope(host) or value.get('scope') != scope(host)
            or value.get('stage') not in ('root', 'opening', 'detail', 'closing')
            or not isinstance(value.get('identities'), dict) or any(key not in ('0', '1', '2', '3')
                or identity not in host._drink_ids for key, identity in value['identities'].items())
            or host.ledger is not None and host.ledger.pending is not None):
        raise ValueError('drink census does not belong to this recovery epoch')
    return value


def with_request(handoff, value):
    result = deepcopy(handoff or {})
    result.setdefault('nonbattle_inventory', {})['drink_inventory'] = deepcopy(value)
    return result


def _frames(proof, excluded=()):
    keys = proof.get('frame_ids')
    if (not isinstance(keys, list) or len(keys) != 1 or not isinstance(keys[0], str)
            or not keys[0].strip() or keys[0] in excluded):
        raise ValueError('drink census requires one independent current frame')
    return keys


def _root(proof, value=None):
    slots = proof.get('slots')
    if (proof.get('kind') != 'root' or not isinstance(slots, list) or len(slots) != 4
            or [row.get('slot') for row in slots] != list(range(4))
            or any(row.get('state') not in ('EMPTY', 'OCCUPIED') for row in slots)
            or any(type(proof.get(key)) is not int for key in ('stamina', 'max_stamina', 'currency'))
            or not 0 <= proof['stamina'] <= proof['max_stamina'] <= 100 or proof['max_stamina'] < 1 or proof['currency'] < 0):
        raise ValueError('drink census root is not a complete current interval')
    if value is not None and ([[row['slot'], row['state']] for row in slots] != value['occupancy']
            or [proof[key] for key in ('stamina', 'max_stamina', 'currency')] != value['resources']):
        raise ValueError('drink census changed resources or occupancy during cancel-only inspection')
    return slots


def next_slot(host, proof):
    value = request(host)
    slots = _root(proof, value)
    known = (value or {}).get('identities', {})
    if any(str(slot) in known and slots[slot]['state'] != 'OCCUPIED' for slot in range(4)):
        raise ValueError('drink census lost a previously inspected occupied slot')
    return next((row['slot'] for row in slots if row['state'] == 'OCCUPIED' and str(row['slot']) not in known), None)


def checkpoint(host, gate, handoff, *, confirmed=None, cancel_unissued=False):
    if host.ownership is None or host.ownership['phase'] != 'recovery': return handoff
    value = request(host, handoff=handoff)
    if gate.pending is None:
        if cancel_unissued and value is not None and value['stage'] in ('opening', 'closing'):
            value = deepcopy(value)
            value['stage'] = 'root' if value['stage'] == 'opening' else 'detail'
            if value['stage'] == 'root': value['active_slot'] = None
            return with_request(handoff, value)
        return with_request(handoff, confirmed) if confirmed is not None else handoff
    pending, page = gate.pending, host._page
    if pending.target.kind not in KINDS: return handoff
    if (page is None or page.observation != pending.source or not isinstance(page.drink_inventory, dict)
            or tuple(_frames(page.drink_inventory)) != page.frame_ids):
        raise ValueError('drink input lacks its exact current physical-slot source')
    proof = page.drink_inventory
    if pending.target.kind == 'recovery_drink_detail':
        slot = next_slot(host, proof)
        if slot is None or pending.target.key != f'drink-detail:{slot}' or proof.get('inspect_slot') != slot:
            raise ValueError('drink inspection is not the unique next unresolved physical slot')
        if value is None:
            value = dict(scope=scope(host), stage='root', identities={}, details={}, active_slot=None,
                occupancy=[[row['slot'], row['state']] for row in proof['slots']],
                resources=[proof[key] for key in ('stamina', 'max_stamina', 'currency')])
        if value['stage'] != 'root': raise ValueError('drink detail opening has no closed prior visit')
        value = deepcopy(value); value.update(stage='opening', active_slot=slot)
    else:
        if (value is None or value['stage'] != 'detail' or proof.get('kind') != 'detail'
                or proof.get('slot') != value['active_slot'] or pending.target.key != f'drink-close:{value["active_slot"]}'
                or proof.get('drink_id') != value['identities'].get(str(value['active_slot']))):
            raise ValueError('drink cancel is not the original current detail')
        value = deepcopy(value); value['stage'] = 'closing'
    value['source_frame_ids'] = list(page.frame_ids)
    return with_request(handoff, value)


def confirm(host, page):
    value, pending, proof = request(host), host.hooks.transition_gate.pending, page.drink_inventory
    if value is None or pending is None or pending.target.kind not in KINDS or not isinstance(proof, dict):
        raise ValueError('drink receipt lacks its original pending inspection')
    if tuple(_frames(proof, value['source_frame_ids'])) != page.frame_ids:
        raise ValueError('drink receipt frame binding changed')
    result = deepcopy(value)
    if pending.target.kind == 'recovery_drink_detail':
        slot = value['active_slot']
        if (value['stage'] != 'opening' or proof.get('kind') != 'detail' or proof.get('slot') != slot
                or proof.get('drink_id') not in host._drink_ids):
            raise ValueError('drink detail does not prove its original physical-slot identity')
        result['identities'][str(slot)] = proof['drink_id']
        result['details'][str(slot)] = deepcopy(proof)
        result['stage'] = 'detail'
    else:
        if value['stage'] != 'closing': raise ValueError('drink cancel did not have an original open panel')
        _root(proof, value)
        result.update(stage='root', active_slot=None)
    return result


def read_detail(context, image, rows, names):
    from .production import _row
    if getattr(image, 'shape', ()) != (1280, 720, 3): raise ValueError('nonbattle drink detail capture is uncalibrated')
    title = _row(rows, r'Pドリンク詳細', (30, 300, 630, 720))
    name = _row(rows, '|'.join(re.escape(value) for value in names), (150, title['box'][1]+60, 490, 90))
    cancel = _row(rows, r'キャンセル|やめる|閉じる|戻る', (40, 1080, 630, 160))
    return dict(drink_id=names[name['text']], target=tuple(cancel['box']), name=name['text'],
        heading=title, title=name, cancel=cancel)


def resolve(host, proof, remainder):
    value = request(host)
    slots = _root(proof, value)
    known = (value or {}).get('identities', {})
    occupied = [row['slot'] for row in slots if row['state'] == 'OCCUPIED']
    if set(known) != {str(slot) for slot in occupied}:
        raise ValueError('post-battle drinks need every current occupied-slot title')
    if host.ledger is None:
        return tuple(DrinkInstance(slot, f'{host.run_id}:recovery:drink:{slot}', known[str(slot)]) for slot in occupied)
    old = host.ledger.state.drinks
    if not old:
        if occupied: raise ValueError('post-battle census contains an unexplained new drink')
        return ()
    handoff = host.handoff or {}
    if (not isinstance(remainder, dict) or remainder.get('schema_version') != 1
            or remainder.get('route_run_id') != host.run_id or remainder.get('game_pid') != host.game_pid
            or remainder.get('handoff_id') != handoff.get('handoff_id') or remainder.get('round_number') != 1
            or remainder.get('terminal_source') != handoff.get('terminal_source')
            or remainder.get('original_ledger_revision') != host.ledger.state.revision):
        raise ValueError('post-battle drinks lack this round verified consumption/remaining source')
    remaining, consumed = remainder.get('remaining_instances'), remainder.get('consumed_instance_ids')
    old_by_id = {drink.instance_id: drink for drink in old}
    if (not isinstance(remaining, list) or not isinstance(consumed, list)
            or any(not isinstance(instance_id, str) or instance_id not in old_by_id for instance_id in consumed)
            or any(not isinstance(row, dict) or row.get('instance_id') not in old_by_id
                or row.get('drink_id') != old_by_id[row['instance_id']].drink_id for row in remaining)
            or Counter([row['instance_id'] for row in remaining] + consumed) != Counter(old_by_id.keys())
            or Counter(known.values()) != Counter(row['drink_id'] for row in remaining)):
        raise ValueError('fresh drink titles disagree with the verified original-instance consumption partition')
    pool = defaultdict(list)
    for row in remaining: pool[row['drink_id']].append(old_by_id[row['instance_id']])
    if any(len(values) != 1 for values in pool.values()):
        raise ValueError('same-name remaining bottles have no unique current physical-instance mapping')
    return tuple(replace(pool[known[str(slot)]][0], slot=slot) for slot in occupied)
