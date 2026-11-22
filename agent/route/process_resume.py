"""Same-save binding proofs/plans and the trusted locked probe entry.

The native task/bootstrap owns endpoint identity and permissions. A plan is
not a grant and cannot be applied by task JSON. Execution stays in the lazy,
separate probe module; importing this module never loads an SDK or writes data.
"""
from __future__ import annotations

import re
import json
from copy import deepcopy
from dataclasses import asdict, dataclass

from .hif import active_hif_role, known_hif_role
from .ledger import HifLedgerState
from .ownership import validate_ownership
from .checkpoint import restore_transition
from .run_budget import remaining_ms, advance_budget, validate_budget

KEY = 'process_resume_probe'


@dataclass(frozen=True)
class SameSaveProof:
    run_id: str
    game_pid: int
    probe_id: str
    started_at_ms: int
    observed_at_ms: int
    identity: tuple[str, str, str]
    state: HifLedgerState
    identity_frame_ids: tuple[str, ...]
    inventory_frame_ids: tuple[str, ...]
    resource_frame_ids: tuple[str, ...]


def _container(record):
    from .cultivation import _budget_container
    container = _budget_container(record.get('handoff') or {})
    if container is None:
        raise ValueError('same-save resume lacks its original cumulative budget')
    return container


def _canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _without_probe(record, *, detached=True):
    # Exclude the child before copying: its original snapshot and growing
    # journal are not part of the business CAS comparison.
    value = dict(record, handoff=dict(record['handoff']))
    context = value['handoff'].get('nonbattle_context')
    if context is not None:
        value['handoff']['nonbattle_context'] = {key: item for key, item in context.items() if key != KEY}
    return deepcopy(value) if detached else value


def _semantic(state):
    value = _canonical(state)
    value.pop('revision')
    value['cards'].sort(key=lambda row: row['instance_id'])
    value['drinks'].sort(key=lambda row: row['slot'])
    return value


def _endpoints(endpoints, game_pid):
    from utils.maa_input import calibrated_live_client_size
    if (not isinstance(endpoints, dict) or not {'game', 'maa', 'agent'} <= set(endpoints)
            or any(key not in ('game', 'maa', 'agent') and not re.fullmatch(r'input_child_[1-9][0-9]*', key)
                for key in endpoints)
            or any(not isinstance(row, dict) or row.get('integrity_rid') != 0x3000
                or type(row.get('pid')) is not int or row['pid'] <= 0 for row in endpoints.values())
            or any(not isinstance(row.get('executable'), str) or not row['executable']
                or type(row.get('created_at_100ns')) is not int or row['created_at_100ns'] <= 0 for row in endpoints.values())
            or len({row['pid'] for row in endpoints.values()}) != len(endpoints)
            or endpoints['game']['pid'] != game_pid
            or calibrated_live_client_size(tuple(endpoints['game'].get('observed_client_size', ()))) is None):
        raise ValueError('same-save process resume requires the complete current High endpoint identity')


def _proof(raw):
    from .ledger import _state
    try:
        value = dict(raw)
        value['state'] = _state(value['state'])
        for key in ('identity', 'identity_frame_ids', 'inventory_frame_ids', 'resource_frame_ids'):
            value[key] = tuple(value[key])
        return SameSaveProof(**value)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError('published same-save process binding lacks its complete original proof') from error


def _verify_binding(binding):
    proof = _proof(binding.get('proof'))
    base, owner = binding.get('base_ledger') or {}, binding.get('from_ownership') or {}
    validate_ownership(owner)
    if (proof.run_id != binding['run_id'] or proof.game_pid != binding['to_game_pid']
            or proof.probe_id != binding['probe_id'] or len(proof.identity) != 3
            or proof.identity[2] != 'HIF' or not known_hif_role(*proof.identity[:2])
            or proof.started_at_ms != binding.get('first_started_at_ms')
            or type(binding.get('published_at_ms')) is not int
            or not 0 <= proof.started_at_ms <= proof.observed_at_ms <= binding['published_at_ms']
            or binding['published_at_ms']-proof.observed_at_ms > 1000
            or owner['game_pid'] != binding['from_game_pid'] or (owner['owner'], owner['phase']) != ('task600', 'nonbattle')
            or binding['ownership_revision'] != owner['revision']+1
            or base.get('schema_version') != 1 or base.get('pending') is not None
            or (base.get('state') or {}).get('revision') != binding['source_ledger_revision']
            or _semantic(asdict(proof.state)) != _semantic(base['state'])):
        raise ValueError('published same-save proof differs from its original business scope')
    _endpoints(binding.get('endpoints'), proof.game_pid)
    batches = (proof.identity_frame_ids, proof.inventory_frame_ids, proof.resource_frame_ids)
    for index, frames in enumerate(batches):
        _frames(frames, game_pid=proof.game_pid, started=proof.started_at_ms,
            observed=proof.observed_at_ms, count=None if index == 1 else 1)
    flat = [key for frames in batches for key in frames]
    if len(set(flat)) != len(flat):
        raise ValueError('published same-save proof reused a page source')
    _fresh_resource_batch(proof, now_ms=binding['published_at_ms'])
    old, published = binding.get('original_budget') or {}, binding.get('published_budget') or {}
    if (old.get('game_pid') != binding['from_game_pid'] or published.get('game_pid') != proof.game_pid
            or any(published.get(key) != old.get(key) for key in ('run_id', 'clock_epoch', 'started_at_ms',
                'windows_used', 'max_windows', 'max_elapsed_ms', 'initial_process_id', 'initial_day', 'initial_phase', 'terminal_source'))
            or old.get('schema_version') == 1 and published.get('legacy_migration', {}).get('original') != old):
        raise ValueError('published same-save binding lacks its original immutable budget source')
    validate_budget(published, run_id=proof.run_id, game_pid=proof.game_pid, epoch=old.get('clock_epoch'),
        now=binding['published_at_ms'], max_windows=64, max_elapsed_ms=21_600_000)
    changes = binding.get('probe_instance_rebinds', [])
    first_identity = binding['endpoints']['game']
    if changes:
        if not isinstance(changes, list) or not isinstance(changes[0], dict) or not isinstance(changes[0].get('from_game_identity'), dict):
            raise ValueError('same-save probe instance history is incomplete')
        first_identity = changes[0]['from_game_identity']
    if published.get('continuations') is not None:
        from .continuation import validate_credit
        credits = published['continuations']
        credit = validate_credit(credits[0], now=binding['published_at_ms'], epoch=old['clock_epoch'])
        initial = credit['initial_budget']
        if (credit['probe_id'] != proof.probe_id or credit['first_started_at_ms'] != proof.started_at_ms
                or (initial.get('legacy_migration', {}).get('original') != old if old['schema_version'] == 1 else initial != old)):
            raise ValueError('published continuation changed its complete predecessor source')
        expected_reference = (_continuation_reference(credit, binding)
            if credit['grant']['actual_game_pid'] != proof.game_pid else None)
        if published.get('continuation_process_binding') != expected_reference:
            raise ValueError('published continuation lacks its exact process-binding reference')
        first_identity = credit['source_probe']['game_identity']
    _probe_instance_identity(dict(game_identity=first_identity, current_game_identity=binding['endpoints']['game'],
        instance_rebinds=changes, probe_id=proof.probe_id, first_started_at_ms=proof.started_at_ms),
        now=binding['published_at_ms'])
    before = (old['last_at_ms']-old['started_at_ms']) if old['schema_version'] == 1 else old['active_elapsed_ms']
    charged_from = old['last_at_ms'] if old.get('accounting_state') == 'active' else proof.started_at_ms
    if published['active_elapsed_ms'] != before + binding['published_at_ms'] - charged_from:
        raise ValueError('published same-save binding excluded actual probe time')


def read_bound_game_pid(shared_record, *, run_id, original_game_pid, current_endpoints=None):
    """Read a published binding chain; historical owner changes grant no input."""
    if (not isinstance(shared_record, dict) or shared_record.get('schema_version') != 3
            or shared_record.get('run_id') != run_id
            or (shared_record.get('ledger') or {}).get('state', {}).get('run_id') != run_id):
        raise ValueError('same-save binding changed its approved run')
    owner = validate_ownership(shared_record['ownership'])
    container = _container(shared_record)
    budget = container['automatic_cultivation']
    history = container.get('process_bindings', [])
    if not isinstance(history, list):
        raise ValueError('same-save process binding history is malformed')
    current, revision = original_game_pid, 0
    for binding in history:
        if (not isinstance(binding, dict) or binding.get('schema_version') != 1
                or binding.get('run_id') != run_id or binding.get('from_game_pid') != current
                or type(binding.get('to_game_pid')) is not int or binding['to_game_pid'] <= 0
                or binding['to_game_pid'] == current or type(binding.get('ownership_revision')) is not int
                or binding['ownership_revision'] <= revision
                or not isinstance(binding.get('probe_id'), str) or not binding['probe_id']
                or type(binding.get('source_ledger_revision')) is not int
                or shared_record['ledger']['state']['revision'] < binding['source_ledger_revision']):
            raise ValueError('same-save process binding history is incomplete or belongs to another scope')
        current, revision = binding['to_game_pid'], binding['ownership_revision']
        _verify_binding(binding)
    if (owner['game_pid'] != current or owner['revision'] < revision
            or budget.get('game_pid') != current or budget.get('run_id') != run_id):
        raise ValueError('current owner or budget does not match the published process binding')
    if history:
        published = history[-1]['published_budget']
        if budget.get('continuations') != published.get('continuations'):
            raise ValueError('current budget erased or reactivated its published continuation')
        if budget.get('continuation_process_binding') != published.get('continuation_process_binding'):
            raise ValueError('current budget changed its proven continuation process binding')
        if (any(budget.get(key) != published.get(key) for key in ('started_at_ms', 'max_windows',
                'max_elapsed_ms', 'initial_process_id', 'initial_day', 'initial_phase', 'clock_epoch'))
                or budget.get('windows_used', -1) < published['windows_used']
                or budget.get('active_elapsed_ms', -1) < published['active_elapsed_ms']):
            raise ValueError('current budget reset its published same-save history')
        if current_endpoints is not None:
            _endpoints(current_endpoints, current)
            if not _same_probe_instance(history[-1]['endpoints']['game'], current_endpoints['game']):
                raise ValueError('current game instance differs from its last complete published binding')
    return current


def _frames(values, *, game_pid, started, observed, count=None):
    if (not isinstance(values, tuple) or not values or count is not None and len(values) != count
            or len(set(values)) != len(values)):
        raise ValueError('same-save proof lacks independent current frame references')
    for key in values:
        match = re.fullmatch(f'route-frame-{game_pid}-([0-9]+)', key) if isinstance(key, str) else None
        if match is None or not started <= int(match[1]) <= observed:
            raise ValueError('same-save proof frame is outside its current process and probe')


def _same_probe_instance(previous, current):
    """Compare persistent process identity in an already calibrated viewport.

    Raw client edges may round within the existing profile between native
    calls. They do not create a new process. Actual geometry remains pinned
    independently for each capture/input scope.
    """
    from utils.maa_input import calibrated_live_client_size

    if not isinstance(previous, dict) or not isinstance(current, dict):
        return False
    geometry = {'observed_client_size', 'reference_client_size', 'calibrated_client_size'}
    if {key: value for key, value in previous.items() if key not in geometry} != {
            key: value for key, value in current.items() if key not in geometry}:
        return False
    profiles = []
    for row in (previous, current):
        profile = calibrated_live_client_size(tuple(row.get('observed_client_size', ())))
        if profile is None:
            return False
        if ('calibrated_client_size' in row and tuple(row['calibrated_client_size']) != profile
                or 'reference_client_size' in row and
                calibrated_live_client_size(tuple(row['reference_client_size'])) != profile):
            return False
        profiles.append(profile)
    return profiles[0] == profiles[1]


def _probe_instance_identity(state, *, now):
    """Current capture identity follows closed instance changes; credits stay historical."""
    identity = state['game_identity']
    previous_at = state['first_started_at_ms']
    changes = state.get('instance_rebinds', [])
    if not isinstance(changes, list):
        raise ValueError('same-save probe instance history is incomplete')
    for change in changes:
        if not isinstance(change, dict):
            raise ValueError('same-save probe instance history is incomplete')
        proof = change.get('previous_proof') or {}
        if (change.get('from_game_identity') != identity
                or not isinstance(change.get('to_game_identity'), dict)
                or change['to_game_identity'] == identity
                or change['to_game_identity'].get('executable') != identity.get('executable')
                or type(change.get('observed_at_ms')) is not int
                or not previous_at <= change['observed_at_ms'] <= now
                or proof.get('game_pid') != identity['pid']
                or proof.get('probe_id') != state['probe_id']
                or proof.get('started_at_ms') != state['first_started_at_ms']):
            raise ValueError('same-save probe instance history changed its closed predecessor')
        identity, previous_at = change['to_game_identity'], change['observed_at_ms']
    current = state.get('current_game_identity', state['game_identity'])
    if current != identity and not _same_probe_instance(identity, current):
        raise ValueError('same-save probe current instance lacks its closed history')
    return identity


def _continuation_reference(credit, binding):
    # A small reference to the existing primary process_bindings proof. The
    # root reader checks it against that complete atomic publication.
    return dict(probe_id=binding['probe_id'], ownership_revision=binding['ownership_revision'],
        from_game_pid=credit['grant']['actual_game_pid'], to_game_pid=binding['to_game_pid'],
        published_at_ms=binding['published_at_ms'])


def _fresh_resource_batch(proof, *, now_ms):
    """Publication uses current resources; old identity/census sources stay old."""
    _frames(proof.resource_frame_ids, game_pid=proof.game_pid, started=proof.started_at_ms,
        observed=proof.observed_at_ms, count=1)
    tick = int(proof.resource_frame_ids[0].rsplit('-', 1)[1])
    if not 0 <= now_ms - tick <= 1000:
        raise ValueError('same-save final current resource frames exceeded their original 1000ms gate')


def reviewed_stopped_budget(record, *, old_pid, plan, clock_epoch, now_ms):
    """Verify the immutable accounting source before any probe navigation."""
    old_budget = _container(record)['automatic_cultivation']
    if old_budget['schema_version'] == 1:
        plan = plan or {}
        budget = deepcopy(plan.get('budget') or {})
        if (plan.get('kind') != 'legacy_active_runtime_migration_preview'
                or plan.get('pause_exclusions_accepted') is not True
                or plan.get('expected_host_record') != _without_probe(record)
                or budget.get('legacy_migration', {}).get('original') != old_budget
                or any(budget.get(key) != old_budget.get(key) for key in ('run_id', 'game_pid', 'clock_epoch',
                    'started_at_ms', 'windows_used', 'max_windows', 'max_elapsed_ms', 'initial_process_id',
                    'process_ids', 'initial_day', 'initial_phase', 'terminal_source'))):
            raise ValueError('same-save process resume lacks its exact original audited budget migration')
    else:
        budget = deepcopy(old_budget)
    validate_budget(budget, run_id=record['run_id'], game_pid=old_pid, epoch=clock_epoch, now=now_ms,
        max_windows=64, max_elapsed_ms=21_600_000)
    if budget['status'] != 'stopped':
        raise ValueError('same-save process resume has no auditable stopped-to-probe boundary')
    return old_budget, budget


def preview_same_save_process_resume(record, proof, *, endpoints, now_ms, clock_epoch,
        original_game_pid, approved_budget_migration=None):
    """Construct a review candidate; retain all original business bytes/data.

    Only the already audited stopped interval may be excluded. Probe time,
    including a crash gap, is charged from its immutable first-start witness.
    This function never writes the host and never refreshes an action TTL.
    """
    original = deepcopy(record)
    old_pid = read_bound_game_pid(original, run_id=original.get('run_id'), original_game_pid=original_game_pid)
    owner = validate_ownership(original['ownership'])
    gate = restore_transition(original['transition'], clock_epoch=clock_epoch)
    if ((owner['owner'], owner['phase']) != ('task600', 'nonbattle') or gate.pending is not None or gate.stopped
            or original['ledger'].get('pending') is not None):
        raise ValueError('same-save process resume cannot replace an unresolved business transaction')
    if (not isinstance(proof, SameSaveProof) or proof.run_id != original['run_id']
            or proof.identity != (*active_hif_role(original), 'HIF') or not isinstance(proof.state, HifLedgerState)
            or type(proof.game_pid) is not int or proof.game_pid <= 0
            or type(proof.started_at_ms) is not int or type(proof.observed_at_ms) is not int
            or type(now_ms) is not int or not 0 <= proof.started_at_ms <= proof.observed_at_ms <= now_ms
            or now_ms-proof.observed_at_ms > 1000 or not re.fullmatch(r'[a-z0-9-]+', proof.probe_id)):
        raise ValueError('same-save process resume lacks its fresh approved identity/probe')
    child = (original.get('handoff') or {}).get('nonbattle_context', {}).get(KEY)
    if child is not None:
        if (not isinstance(child, dict) or child.get('schema_version') != 1 or child.get('status') != 'complete'
                or child.get('probe_id') != proof.probe_id or child.get('first_started_at_ms') != proof.started_at_ms):
            raise ValueError('same-save process resume probe is unresolved or belongs to another source')
        child_gate = restore_transition(child.get('transition'), clock_epoch=clock_epoch)
        if child_gate.pending is not None or child_gate.stopped:
            raise ValueError('same-save process resume cannot publish a pending probe')
    _endpoints(endpoints, proof.game_pid)
    if proof.game_pid == old_pid:
        raise ValueError('same-save process resume requires a separately verified new instance')
    batches = (proof.identity_frame_ids, proof.inventory_frame_ids, proof.resource_frame_ids)
    for index, frames in enumerate(batches):
        _frames(frames, game_pid=proof.game_pid, started=proof.started_at_ms,
            observed=proof.observed_at_ms, count=None if index == 1 else 1)
    flat = [key for batch in batches for key in batch]
    if len(set(flat)) != len(flat):
        raise ValueError('same-save identity, inventory and resources reused a page source')
    _fresh_resource_batch(proof, now_ms=now_ms)
    if _semantic(asdict(proof.state)) != _semantic(original['ledger']['state']):
        raise ValueError('current complete save differs from the original ledger; do not amend the ledger')
    old_budget, budget = reviewed_stopped_budget(original, old_pid=old_pid, plan=approved_budget_migration,
        clock_epoch=clock_epoch, now_ms=now_ms)
    if not budget['last_at_ms'] <= proof.started_at_ms:
        raise ValueError('same-save process resume has no auditable stopped-to-probe boundary')
    binding = dict(schema_version=1, run_id=proof.run_id, probe_id=proof.probe_id,
        from_game_pid=old_pid, to_game_pid=proof.game_pid, ownership_revision=owner['revision']+1,
        source_ledger_revision=original['ledger']['state']['revision'], proof=_canonical(asdict(proof)),
        base_ledger=deepcopy(original['ledger']), from_ownership=deepcopy(owner),
        endpoints=deepcopy(endpoints), first_started_at_ms=proof.started_at_ms, published_at_ms=now_ms)
    if child and child.get('instance_rebinds'):
        if not _same_probe_instance(_probe_instance_identity(child, now=now_ms), _canonical(endpoints['game'])):
            raise ValueError('same-save proof does not belong to the current probe instance')
        binding['probe_instance_rebinds'] = deepcopy(child['instance_rebinds'])
    # End the already acknowledged pause at the FIRST probe start, rather
    # than publication. No cultivation execution window is minted/renewed.
    credit = child.get('continuation') if child else None
    if child and child.get('engineering_continuation') is not None and credit is None:
        raise ValueError('same-save engineering continuation lost its complete first credit')
    if child and child.get('final_engineering_continuation') is not None and child.get('engineering_continuation') is None:
        raise ValueError('same-save final engineering continuation lost its complete second credit')
    if credit is not None:
        from .continuation import probe_remaining
        probe_remaining(budget, child, now=now_ms, epoch=clock_epoch)
        extension = child.get('engineering_continuation')
        final = child.get('final_engineering_continuation')
        budget = deepcopy((final or extension or credit)['predecessor_budget'])
        budget['game_pid'] = proof.game_pid
        budget['continuations'] = ([deepcopy(credit)] + ([deepcopy(extension)] if extension is not None else [])
            + ([deepcopy(final)] if final is not None else []))
        if credit['grant']['actual_game_pid'] != proof.game_pid:
            budget['continuation_process_binding'] = _continuation_reference(credit, binding)
    elif budget['accounting_state'] == 'paused':
        budget['pause_receipts'].append(dict(from_ms=budget['paused_at_ms'], to_ms=proof.started_at_ms,
            evidence=dict(kind='formal_runtime_stopped_checkpoint', process_id=budget['last_process_id'],
                reason='same_save_process_probe_started', windows_used=budget['windows_used'], probe_id=proof.probe_id)))
        budget.update(last_at_ms=proof.started_at_ms, paused_at_ms=None, accounting_state='active')
    budget['status'] = 'running'
    budget = advance_budget(budget, now=now_ms, process_id=endpoints['agent']['pid'], opening=False)
    candidate = deepcopy(original)
    candidate['ownership'].update(game_pid=proof.game_pid, revision=owner['revision']+1)
    target = _container(candidate)
    target['process_bindings'] = [*target.get('process_bindings', []), binding]
    budget['game_pid'] = proof.game_pid
    binding.update(original_budget=deepcopy(old_budget), published_budget=deepcopy(budget))
    target['automatic_cultivation'] = budget
    read_bound_game_pid(candidate, run_id=proof.run_id, original_game_pid=original_game_pid)
    validate_budget(budget, run_id=proof.run_id, game_pid=proof.game_pid, epoch=clock_epoch, now=now_ms,
        max_windows=64, max_elapsed_ms=21_600_000)
    return dict(kind='same_save_process_resume_preview', changed=True, game_pid=proof.game_pid,
        binding_revision=owner['revision']+1, expected_host_record=original, candidate=candidate,
        requires_review_before_application=True, remaining_ms=remaining_ms(budget, now_ms))


def resume_same_save_process(context, *, path, run_id, original_game_pid, endpoint,
        authorise_capture, approve_navigation, approved_budget_migration, evidence_sink=None, clock=None,
        capture_endpoint=None, authorise_probe_state=None, authorise_continuation=None, with_probe_record=None,
        authorise_engineering_continuation=None, authorise_final_engineering_continuation=None):
    """Trusted native-task entry; no values from Task JSON can grant this call."""
    from .process_resume_probe import ProcessResumeProbe
    return ProcessResumeProbe(context, path=path, run_id=run_id, original_game_pid=original_game_pid,
        endpoint=endpoint, authorise_capture=authorise_capture, approve_navigation=approve_navigation,
        approved_budget_migration=approved_budget_migration, evidence_sink=evidence_sink, clock=clock,
        capture_endpoint=capture_endpoint, authorise_probe_state=authorise_probe_state,
        authorise_continuation=authorise_continuation, with_probe_record=with_probe_record,
        authorise_engineering_continuation=authorise_engineering_continuation,
        authorise_final_engineering_continuation=authorise_final_engineering_continuation).run(publish=True)
