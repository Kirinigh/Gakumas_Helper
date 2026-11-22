"""Read bottle consumption from this battle's dispatch and settlement records."""
from __future__ import annotations

import re
import json
import importlib
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
from contextlib import contextmanager


@contextmanager
def _native_readers():
    # Runtime is exclusively a JSON data path. Readers belong to the shipped
    # agent package and use relative imports for their business dependencies.
    from .hif_runtime_sources import hif_runtime_sources

    with hif_runtime_sources() as directory:
        modules = [importlib.import_module('.hif_native.' + name, __package__) for name in (
            'native_opening_root', 'native_settlement_resume')]
        if any(Path(module.__file__).resolve().parent != directory for module in modules):
            raise ValueError('terminal drink reader source is not the repository implementation')
        yield modules


def _read(runtime, name):
    if not isinstance(name, str) or re.fullmatch(r'[0-9]+-[a-z0-9-]+', name) is None:
        raise ValueError('invalid terminal drink source identity')
    path = (runtime / f'result-{name}.json').resolve(strict=True)
    if path.parent != runtime.resolve(strict=True):
        raise ValueError('terminal drink source escapes its runtime')
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('id') != name:
        raise ValueError('terminal drink source identity changed')
    return value


def _chain(runtime, terminal, origin, opening_reader):
    rows, seen = [], set()
    row = terminal
    while True:
        name = row['id']
        if name in seen:
            raise ValueError('terminal drink source chain is cyclic')
        seen.add(name)
        if (row.get('handoff_id') != origin['handoff']['handoff_id']
                or row.get('endpoints', {}).get('game', {}).get('pid') != origin['ownership']['game_pid']):
            raise ValueError('terminal drink source crosses its original battle')
        rows.append(row)
        # Component-only evidence is rechecked by the native loader; it does
        # not replace the original submitted action as the business parent.
        keys = [key for key in opening_reader.PARENT_FIELDS if key != 'native_pending_components_source']
        keys.append('native_support_source')
        parents = {row[key] for key in keys if isinstance(row.get(key), str) and row[key]}
        # A restored observation can publish another accepted opening event.
        # Its original business predecessor still owns the consumption history.
        if not parents and opening_reader.has_native_opening_root(row):
            return list(reversed(rows))
        if len(parents) != 1:
            raise ValueError('terminal drink business predecessor is missing or ambiguous')
        row = _read(runtime, parents.pop())


def _submitted(row):
    return any(event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0)
        for event in row.get('live_events', ()))


def _align_instances(instances, observed_slots, *, after_consumption):
    """Keep original order only when the verified native slots prove that order."""
    actual = tuple(tuple(pair) for pair in observed_slots)
    ordered = sorted(instances, key=lambda row: row['slot'])
    expected = tuple((row['slot'], row['drink_id']) for row in ordered)
    compacted = tuple(enumerate(row['drink_id'] for row in ordered))
    if actual != expected and not (after_consumption and actual == compacted):
        raise ValueError('verified battle slots do not preserve original bottle instances')
    return [dict(row, slot=slot) for row, (slot, _) in zip(ordered, actual, strict=True)]


def _accepted_drink_slots(event, action, old):
    from dataclasses import asdict

    from .opening_observer import valid_opening_frame_ids
    from .high_score.opening import observed_hif_drink_slots, verify_hif_consumed_slots
    from .high_score.hif_rule_progress import rule_drink_slots

    component = (event.get('components') or {}).get('drinks')
    reported = event.get('drink_slots')
    if component is not None:
        observed = observed_hif_drink_slots(component)
    elif reported is not None:
        observed = tuple(map(tuple, reported))
    else:
        return None
    if action is None:
        return observed
    if component is None:
        progress = event.get('rule_progress') or {}
        proof = progress.get('action_confirmation') or {}
        if (progress.get('accepted') is not True or progress.get('action') != asdict(action)
                or proof.get('slot') != action.source_slot or proof.get('drink_id') != action.drink_id
                or proof.get('stable') is not True or not valid_opening_frame_ids(proof.get('frame_ids'))
                or proof['frame_ids'] != progress.get('core', {}).get('frame_ids')
                or proof.get('kind') not in ('drink_slot_empty', 'drink_slots_compacted')
                or proof['kind'] == 'drink_slot_empty' and proof.get('empty_observations') != len(proof['frame_ids'])
                or rule_drink_slots(SimpleNamespace(drink_slots=old), action, proof) != observed):
            raise ValueError('terminal drink lacks the confirmed slot change')
    return verify_hif_consumed_slots(old, observed, action, allow_compaction=True)


def _replay(runtime, rows, origin, readers):
    from utils.runtime_paths import resolve_runtime_path

    from .hif_visual import HifLiveRunContext
    from .high_score.contracts import ActionKind

    opening_reader, loader = readers
    module = SimpleNamespace(HifLiveRunContext=HifLiveRunContext,
        CATALOG_PATH=resolve_runtime_path(Path('assets/data/skill_cards.csv'), Path('data/skill_cards.csv')))
    initial, _ = opening_reader.load_native_opening_root(rows[0], module)
    saved = origin['ledger']['state']
    instances = deepcopy(saved['drinks'])
    if (saved.get('drink_count') != len(instances)
            or len({row['instance_id'] for row in instances}) != len(instances)
            or len({row['slot'] for row in instances}) != len(instances)
            or any(set(row) != {'slot', 'instance_id', 'drink_id'}
                or type(row['slot']) is not int or not 0 <= row['slot'] < 4
                or not isinstance(row['instance_id'], str) or not row['instance_id']
                or not isinstance(row['drink_id'], str) or not row['drink_id'] for row in instances)):
        raise ValueError('original route bottle instances are incomplete')
    instances = _align_instances(instances, initial.drink_slots, after_consumption=False)
    if (initial.run_id, initial.revision) != (
            origin['handoff']['live_context']['run_id'], origin['handoff']['live_context']['revision']):
        raise ValueError('terminal drink opening belongs to another live epoch')

    # A zero-input result reread is not itself a final dispatch.
    dispatched = [row for row in rows if _submitted(row)]
    if not dispatched:
        raise ValueError('terminal drink proof lacks an actual final dispatch')
    final = dispatched[-1]
    tail = rows[rows.index(final) + 1:]
    if any(row.get('input_count') != 0 or row.get('native_input_dispatches') for row in tail):
        raise ValueError('terminal reread has unaccounted input after the final dispatch')
    action, _ = loader.decoded_dispatched_action(final, None)
    if action.kind not in (ActionKind.PLAY_CARD, ActionKind.END_TURN):
        raise ValueError('terminal drink proof needs a verified final card or end turn')

    consumed, receipts, pending = [], [], None
    for row in rows:
        drink_events = [event for event in row.get('live_events', ())
            if event.get('event') == 'live_joint_action' and event.get('execution_submit_count', 0)]
        if pending is not None and _submitted(row):
            raise ValueError('terminal drink dispatch has no accepted post-use settlement')
        slots = tuple((item['slot'], item['drink_id']) for item in instances)
        if drink_events:
            if len(drink_events) != 1:
                raise ValueError('terminal drink source has ambiguous consumption')
            # Only the original physical slots and catalog names are needed to
            # verify the native use-button dispatch. Never rerun card simulation.
            bottle_context = SimpleNamespace(drink_slots=slots, kernel=initial.kernel)
            used, _ = loader.decoded_dispatched_action(row, bottle_context)
            if used.kind is not ActionKind.USE_DRINK or (used.source_slot, used.drink_id) not in slots:
                raise ValueError('terminal drink dispatch is not a verified bottle action')
            pending = row, used
        for event in row.get('live_events', ()):
            if event.get('event') != 'live_committed_state' or event.get('accepted') is not True:
                continue
            source, used = pending or (None, None)
            observed = _accepted_drink_slots(event, used, slots)
            if observed is None:
                continue
            if pending is None:
                instances = _align_instances(instances, observed, after_consumption=False)
                continue
            item = next(item for item in instances if item['slot'] == used.source_slot)
            instances.remove(item)
            consumed.append(item['instance_id'])
            receipts.append(dict(source_record=source['id'], instance_id=item['instance_id'],
                drink_id=item['drink_id'], source_slot=used.source_slot))
            instances = _align_instances(instances, observed, after_consumption=True)
            slots = observed
            pending = None
    if pending is not None:
        raise ValueError('terminal drink dispatch has no accepted post-use settlement')
    return [dict(instance_id=item['instance_id'], drink_id=item['drink_id']) for item in instances], consumed, receipts


def load_verified_terminal_drink_remainder(runtime, shared_record, report=None):
    """Return historical remaining instances; caller must observe current slots."""
    from route.terminal_source import verify_bound_terminal_return

    from .hif_battle_entry import verify_prepared_terminal_return

    runtime = Path(runtime)
    record = deepcopy(shared_record)
    handoff = record['handoff']
    stored = _read(runtime, handoff.get('terminal_source'))
    if report is not None:
        provided, disk = deepcopy(report), deepcopy(stored)
        # The sealed disk report predates successful publication. This sole
        # marker may be present in the in-memory return; the shared proof below
        # independently verifies that publication really happened.
        provided.pop('returned_to_nonbattle', None)
        disk.pop('returned_to_nonbattle', None)
        if provided != disk:
            raise ValueError('provided terminal report differs from its original source')
    if handoff.get('terminal_return_origin') is not None:
        verify_bound_terminal_return(run_id=record['run_id'], ownership=record['ownership'], handoff=handoff, report=stored)
        origin = handoff['terminal_return_origin']['record']
    else:
        verify_prepared_terminal_return(None, record, stored)
        origin = record
    with _native_readers() as readers:
        rows = _chain(runtime, stored, origin, readers[0])
        remaining, consumed, receipts = _replay(runtime, rows, origin, readers)
    return dict(schema_version=1, terminal_source=stored['id'], route_run_id=origin['run_id'],
        handoff_id=origin['handoff']['handoff_id'], round_number=origin['handoff']['round_number'],
        game_pid=origin['ownership']['game_pid'], original_ledger_revision=origin['handoff']['ledger_revision'],
        remaining_instances=remaining, consumed_instance_ids=consumed, consumption_sources=receipts,
        source_records=[row['id'] for row in rows], scope='verified_battle_history_not_current_nonbattle_observation')
