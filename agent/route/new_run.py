"""Allocate one cultivation with the current task and its independent AP budget."""
import re
import json
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path

from .engine import TransitionGate, has_single_frame_id
from .ownership import write_hif_record, locked_hif_record, validate_ownership
from .checkpoint import restore_transition, transition_checkpoint
from .closing_state import completed_memory

def authorised_interrupted_start(record, credit):
    """Release builds do not carry a one-time interrupted-run grant."""
    return False


def additional_start_credit(previous_run_id, authorization_id):
    """One human-authorised additional run, retained in the existing start state."""
    import uuid

    if any(not isinstance(value, str) or re.fullmatch(r'[a-zA-Z0-9-]{1,100}', value) is None
            for value in (previous_run_id, authorization_id)):
        raise ValueError('additional cultivation identity is invalid')
    from .battle_policy import STRONG_PROFILE
    return dict(authorization_id=authorization_id, previous_run_id=previous_run_id,
        run_id='hif-' + uuid.uuid5(uuid.NAMESPACE_URL, authorization_id + ':' + previous_run_id).hex,
        ap_cost=20, maximum_cultivations=1, policy_profile=STRONG_PROFILE, status='available')


def validate_additional_start(credit, *, run_id, previous_run_id, status=None):
    if not isinstance(credit, dict):
        raise ValueError('additional cultivation lacks its original credit')
    expected = additional_start_credit(previous_run_id, credit.get('authorization_id'))
    expected['status'] = credit.get('status')
    if (credit != expected or credit['run_id'] != run_id
            or credit['status'] not in ('available', 'pending', 'consumed')
            or status is not None and credit['status'] != status):
        raise ValueError('additional cultivation credit changed or was already used')
    return credit


def update_additional_start(state, status, *, cancelled_unissued=False):
    credit = state.get('additional_start')
    if credit is not None:
        validate_additional_start(credit, run_id=state['run_id'], previous_run_id=state['previous_run_id'])
        order = ('available', 'pending', 'consumed')
        unissued = (cancelled_unissued is True and status == 'available' and credit['status'] == 'pending'
            and state.get('status') == 'available' and bool(state.get('cancelled_unissued_requests')))
        if status not in order or order.index(status) < order.index(credit['status']) and not unissued:
            raise ValueError('additional cultivation credit cannot be restored')
        state['additional_start'] = dict(credit, status=status)


def start_record(handoff):
    """Handover retains this same state inside nonbattle_context."""
    for _ in range(20):
        if not isinstance(handoff, dict):
            return None
        if 'cultivation_start' in handoff:
            value = handoff['cultivation_start']
            if not isinstance(value, dict):
                raise ValueError('new cultivation start state is invalid')
            return value
        handoff = handoff.get('nonbattle_context')
    raise ValueError('new cultivation history nesting is invalid')


def _memory_stats(values):
    if (isinstance(values, (tuple, list)) and len(values) == 3
            and all(type(value) is int and value >= 0 for value in values)):
        return tuple(values)
    return None


def start_memory_target(host):
    """The first selected Vo/Da/Vi observed at this cultivation's confirmation."""
    state = start_record(host.handoff)
    if state is None:
        return None
    if state.get('run_id') != host.run_id:
        raise ValueError('qualifier memory belongs to another cultivation')
    binding = state.get('selected_memory')
    if binding is None:
        return None
    if (not isinstance(binding, dict) or binding.get('run_id') != host.run_id
            or _memory_stats(binding.get('stats')) is None or not has_single_frame_id(binding.get('frame_ids'))):
        raise ValueError('selected qualifier memory is not bound to this cultivation')
    return tuple(binding['stats'])


def retain_start_memory(host, page, frames):
    """Keep the first real confirmation without changing any pending navigation."""
    target = start_memory_target(host)
    state = start_record(host.handoff)
    if (target is not None or state is None or state.get('status') != 'available'
            or page.get('phase') != 'start_confirmation'):
        return target
    from .hif import known_hif_role
    from .start_transaction import _budget
    _budget(host)
    stats = _memory_stats(page.get('memory_stats'))
    role = tuple(page.get('identity', ()))
    if stats is None or len(role) != 2 or not known_hif_role(*role):
        return None
    # The active host holds the record lock. Observing a destination can run
    # before its pending navigation is confirmed, so update only this field.
    if not host._active:
        return None
    stored = json.loads(host.path.read_text(encoding='utf-8'))
    original = start_record(stored.get('handoff'))
    if stored.get('run_id') != host.run_id or original != state:
        raise ValueError('start budget changed before binding its selected qualifier memory')
    snapshot = dict(run_id=host.run_id, stats=list(stats), frame_ids=list(frames))
    original['selected_memory'] = snapshot
    write_hif_record(host.path, stored)
    state['selected_memory'] = snapshot
    return stats


def matches_start_memory(host, page):
    target = start_memory_target(host)
    if target is None or page.get('memory_matches') is not True:
        return False
    return _memory_stats(page.get('memory_expected')) == target


def initial_native_start(state):
    if 'start_origin' not in state:
        return False
    from .cultivation_factory import validate_creation_task
    creation = validate_creation_task(state.get('creation_task'), game_pid=state['game_pid'])
    if (state['start_origin'] != 'native_produce' or creation['entry'] != 'Produce'
            or any(key in state for key in ('previous_run_id', 'previous_record', 'previous_disposition', 'additional_start'))):
        raise ValueError('first cultivation must originate from its current native Produce task')
    return True


def _start_candidate(*, run_id, game_pid, epoch, profile, creation_task, customization_selection, **source):
    from .hif import PRESET_ID

    start = dict(schema_version=1, run_id=run_id, game_pid=game_pid, preset=PRESET_ID,
        policy_profile=profile, maximum_cultivations=1, ap_cost=20, status='available', **source)
    if creation_task is not None:
        start['creation_task'] = deepcopy(creation_task)
    if customization_selection is not None:
        from .customization_runtime import _payload
        start['customization_selection'] = _payload(customization_selection)
    return dict(schema_version=3, run_id=run_id, transition=transition_checkpoint(TransitionGate(), epoch),
        ledger=None, ownership=dict(revision=1, owner='task600', phase='recovery', game_pid=game_pid),
        handoff=dict(cultivation_start=start))


def begin_authorised_new_run(path, *, run_id, previous_run_id, game_pid,
        terminal_report, authorise_new_run, clock, battle_policy=None, creation_task=None,
        additional_start=None, customization_selection=None):
    """No UI input; preserve the completed record before atomic active rotation.

    The caller binds the human's one-run permission independently of task JSON.
    Persistent budget data constrains that permission; it never grants input.
    """
    from .battle_policy import FAST_PROFILE, retained_profile, selected_profile
    if creation_task is not None:
        from .cultivation_factory import validate_creation_task
        validate_creation_task(creation_task, game_pid=game_pid)
    profile = selected_profile(battle_policy)
    if additional_start is not None:
        validate_additional_start(additional_start, run_id=run_id,
            previous_run_id=previous_run_id, status='available')
        if profile != additional_start['policy_profile']:
            raise ValueError('additional cultivation requires its explicit strong policy')
    path = Path(path)
    if (not callable(authorise_new_run) or authorise_new_run() is not True
            or not isinstance(run_id, str) or not isinstance(previous_run_id, str)
            or not re.fullmatch(r'[a-zA-Z0-9-]{1,100}', run_id)
            or not re.fullmatch(r'[a-zA-Z0-9-]{1,100}', previous_run_id)
            or run_id == previous_run_id or type(game_pid) is not int or game_pid <= 0):
        raise ValueError('new cultivation requires the approved independent one-run scope')
    with locked_hif_record(path):
        raw = json.loads(path.read_text(encoding='utf-8'))
        if raw.get('run_id') == run_id:
            prior = start_record(raw.get('handoff'))
            if (prior is None or prior.get('run_id') != run_id or prior.get('game_pid') != game_pid
                    or prior.get('previous_run_id') != previous_run_id):
                raise ValueError('existing new cultivation has different start provenance')
            retained_profile(raw['handoff'], selection=battle_policy)
            if creation_task is not None and prior.get('creation_task') != creation_task:
                raise ValueError('existing new cultivation belongs to its original creation task')
            if additional_start is not None:
                actual = prior.get('additional_start')
                validate_additional_start(actual, run_id=run_id, previous_run_id=previous_run_id)
                if dict(actual, status='available') != additional_start:
                    raise ValueError('existing cultivation has a different additional credit')
            return prior
        interrupted = authorised_interrupted_start(raw, additional_start)
        if (raw.get('schema_version') != 3 or raw.get('run_id') != previous_run_id
                or set(raw) != {'schema_version', 'run_id', 'transition', 'ledger', 'ownership', 'handoff'}
                or not isinstance(raw.get('ledger'), dict) or not interrupted and raw['ledger'].get('pending') is not None
                or (start_record(raw.get('handoff')) is not None and (additional_start is None
                    or start_record(raw['handoff']).get('status') != 'complete'))):
            raise ValueError('one new cultivation cannot replace an active or already-budgeted run')
        owner = validate_ownership(raw['ownership'])
        if not interrupted:
            gate = restore_transition(raw['transition'], clock_epoch=clock.epoch)
            host = SimpleNamespace(run_id=previous_run_id, game_pid=game_pid,
                ownership=owner, handoff=raw['handoff'])
            memory = completed_memory(host, terminal_report)
            closing = raw['handoff'].get('closing', {})
            home = closing.get('home', {})
            keys = home.get('frame_ids', ())
            if (owner['owner'] != 'task600' or owner['phase'] != 'recovery' or owner['game_pid'] != game_pid
                    or gate.pending is not None or gate.stopped or memory is None or closing.get('state') != 'complete'
                    or not has_single_frame_id(keys) or keys[0] in (*memory['source_frame_ids'], *memory['receipt_frame_ids'])
                    or type(home.get('revision')) is not int or home['revision'] <= memory['receipt_revision']
                    or type(home.get('ap')) is not int or type(home.get('max_ap')) is not int
                    or not 0 <= home['ap'] <= home['max_ap'] or home['max_ap'] <= 0):
                raise ValueError('new cultivation requires the original complete memory and home receipt')
        archive = path.parent / ('interrupted' if interrupted else 'completed') / (previous_run_id + '.json')
        if archive.exists():
            if json.loads(archive.read_text(encoding='utf-8')) != raw:
                raise ValueError('completed cultivation archive differs; no overwrite')
        else:
            write_hif_record(archive, raw, create=True)
        from .hif import PRESET_ID

        start = dict(schema_version=1, run_id=run_id, game_pid=game_pid,
            previous_run_id=previous_run_id, previous_record=str(archive.resolve()),
            preset=PRESET_ID, policy_profile=profile or FAST_PROFILE,
            maximum_cultivations=1, ap_cost=20, status='available')
        if interrupted:
            start['previous_disposition'] = 'interrupted_external_unverified'
        if creation_task is not None:
            from copy import deepcopy
            start['creation_task'] = deepcopy(creation_task)
        if additional_start is not None:
            from copy import deepcopy
            start['additional_start'] = deepcopy(additional_start)
        if customization_selection is not None:
            from .customization_runtime import _payload
            start['customization_selection'] = _payload(customization_selection)
        candidate = dict(schema_version=3, run_id=run_id,
            transition=transition_checkpoint(TransitionGate(), clock.epoch), ledger=None,
            ownership=dict(revision=1, owner='task600', phase='recovery', game_pid=game_pid),
            handoff=dict(cultivation_start=start))
        if authorise_new_run() is not True:
            raise ValueError('new cultivation scope was revoked before record rotation')
        write_hif_record(path, candidate)
        return start
