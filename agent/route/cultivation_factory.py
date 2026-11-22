"""Native task identity and once-only cultivation allocation; no game input.

Only the trusted product callback constructs the native task binding.
"""
from __future__ import annotations

import re
import json
import uuid
from copy import deepcopy
from pathlib import Path
from dataclasses import asdict, dataclass

from .hif import PRESET_ID, HifConfig
from .new_run import start_record, begin_authorised_new_run
from .battle_policy import retained_profile
from .process_resume import SameSaveProof, _proof, _frames, _semantic, _endpoints, _fresh_resource_batch


def validate_creation_task(value, *, game_pid=None):
    if (not isinstance(value, dict) or set(value) != {'schema_version', 'entry', 'task_id', 'maa', 'game'}
            or value['schema_version'] != 1 or type(value['schema_version']) is not int
            or value['entry'] not in ('Produce', 'HifAcceptance')
            or type(value['task_id']) is not int or value['task_id'] <= 0):
        raise ValueError('cultivation creation task is incomplete')
    for key in ('maa', 'game'):
        row = value[key]
        if (not isinstance(row, dict) or set(row) != {'pid', 'executable', 'created_at_100ns'}
                or type(row['pid']) is not int or row['pid'] <= 0
                or type(row['created_at_100ns']) is not int or row['created_at_100ns'] <= 0
                or not isinstance(row['executable'], str) or not row['executable']):
            raise ValueError('cultivation creation process identity is incomplete')
    if (value['maa']['pid'] == value['game']['pid']
            or game_pid is not None and value['game']['pid'] != game_pid):
        raise ValueError('cultivation creation task has a different game')
    return value


class NativeCultivationTask:
    """One real registered callback, independent of task JSON booleans."""
    def __init__(self, context, argv, *, expected_maa, endpoint=None, entry='Produce', runtime=None):
        from maa.context import Context
        from maa.custom_action import CustomAction
        node, action = ('ProduceHifAdvanceEntry', 'ProduceHifAdvance') if entry == 'Produce' else ('HifAcceptance', 'HifAcceptance')
        if (entry not in ('Produce', 'HifAcceptance')
                or not isinstance(context, Context) or not isinstance(argv, CustomAction.RunArg)
                or argv.node_name != node or argv.custom_action_name != action
                or argv.task_detail.entry != entry or argv.task_detail.status.running is not True
                or type(argv.task_detail.task_id) is not int or argv.task_detail.task_id <= 0
                or endpoint is not None and not callable(endpoint)):
            raise ValueError('new cultivation requires the real native Produce callback')
        from .bootstrap import _endpoint, _PinnedContext, _instance_endpoint
        self.context, self.argv = _PinnedContext(context), argv
        self.expected_maa = Path(expected_maa)
        self.runtime = Path(runtime) if runtime is not None else self.expected_maa.parent
        self.endpoint = endpoint if endpoint is not None else lambda: _instance_endpoint(
            _endpoint(self.context, expected_maa=self.expected_maa))
        self.active = True
        self.entry = entry
        self.task_id = argv.task_detail.task_id
        self.original = self.current()

    def current(self):
        if not self.active:
            raise ValueError('cultivation task binding is closed')
        tasker, job = self.context.tasker, self.context.get_task_job()
        detail = tasker.get_task_detail(self.task_id)
        if (tasker.stopping or tasker.running is not True
                or job.job_id != self.task_id or job.status.running is not True
                or detail is None or detail.task_id != self.task_id
                or detail.entry != self.entry or detail.status.running is not True):
            raise ValueError('cultivation native task is no longer current')
        endpoints = self.endpoint()
        _endpoints(endpoints, endpoints['game']['pid'])
        if Path(endpoints['maa']['executable']).resolve() != self.expected_maa.resolve():
            raise ValueError('cultivation native Maa executable changed')
        fields = ('pid', 'executable', 'created_at_100ns')
        value = dict(schema_version=1, entry=self.entry, task_id=self.task_id,
            maa={key: endpoints['maa'][key] for key in fields},
            game={key: endpoints['game'][key] for key in fields})
        if hasattr(self, 'original') and value != self.original:
            raise ValueError('cultivation native process instance changed')
        return validate_creation_task(value)

    def close(self):
        self.active = False

    def capture_current(self):
        """Normal Maa screenshot only; this cannot dispatch an input action."""
        self.current()
        job = self.context.tasker.controller.post_screencap().wait()
        if job.succeeded is not True:
            raise ValueError('cultivation current-save observation failed')
        image = job.get()
        self.current()
        return image


@dataclass(frozen=True)
class CultivationBindingPlan:
    run_id: str
    allocated: bool
    additional_ap: int
    creation_task: dict | None
    current_task: dict
    needs_same_save_proof: bool
    observe_only: bool
    process_migrations: tuple[dict, ...] = ()


def completed_terminal_report(record, runtime):
    """Load the original explicit terminal pointer; never guess a latest file."""
    from types import SimpleNamespace

    from .cultivation import _budget_container
    from .closing_state import completed_memory
    container = _budget_container(record.get('handoff')) or {}
    source = (container.get('automatic_cultivation') or {}).get('terminal_source')
    if not isinstance(source, str) or re.fullmatch(r'[0-9]+-[a-z0-9-]+', source) is None:
        raise ValueError('cultivation factory needs the original completed terminal source')
    report = json.loads((Path(runtime) / ('result-' + source + '.json')).read_bytes())
    host = SimpleNamespace(run_id=record['run_id'], game_pid=record['ownership']['game_pid'],
        ownership=record['ownership'], handoff=record['handoff'])
    if report.get('id') != source or completed_memory(host, report) is None:
        raise ValueError('cultivation factory terminal source differs from its completed memory')
    return report


def _verified_task_receipt(raw, receipt, current, creation):
    try:
        from .hif import active_hif_role
        if (not isinstance(receipt, dict)
                or set(receipt) not in ({'schema_version', 'run_id', 'creation_task', 'current_task', 'source_revision', 'proof'},
                    {'schema_version', 'run_id', 'creation_task', 'current_task', 'source_revision', 'proof', 'previous'})
                or type(receipt['schema_version']) is not int or receipt['schema_version'] != 1
                or receipt['current_task'] != current or receipt['creation_task'] != creation
                or receipt['run_id'] != raw['run_id'] or type(receipt['source_revision']) is not int
                or not 0 <= receipt['source_revision'] <= raw['ledger']['state']['revision']):
            return False
        proof = _proof(receipt['proof'])
        if (proof.run_id != raw['run_id'] or proof.game_pid != current['game']['pid']
                or proof.identity != (*active_hif_role(raw), 'HIF')
                or proof.state.run_id != raw['run_id'] or proof.state.revision != receipt['source_revision']
                or not 0 <= proof.started_at_ms <= proof.observed_at_ms):
            return False
        batches = (proof.identity_frame_ids, proof.inventory_frame_ids, proof.resource_frame_ids)
        for index, frames in enumerate(batches):
            _frames(frames, game_pid=proof.game_pid, started=proof.started_at_ms,
                observed=proof.observed_at_ms, count=None if index == 1 else 1)
        flat = [key for frames in batches for key in frames]
        return len(flat) == len(set(flat))
    except (ValueError, TypeError, KeyError, AttributeError):
        return False


def _factory_bound_game(record, *, now):
    """Retain creation provenance while following explicit process bindings."""
    from .cultivation import _budget_container
    from .continuation import execution_process_migrations

    start = start_record(record.get('handoff'))
    if not isinstance(start, dict) or type(start.get('game_pid')) is not int or start['game_pid'] <= 0:
        raise ValueError('cultivation process binding lost its original start')
    creation = validate_creation_task(start.get('creation_task'), game_pid=start['game_pid'])
    budget = (_budget_container(record.get('handoff')) or {}).get('automatic_cultivation')
    migrations = execution_process_migrations(budget, now=now) if isinstance(budget, dict) else []
    game = creation['game']
    for migration in migrations:
        if (migration['from_game'] != game
                or migration['handoff']['policy_profile'] != retained_profile(record['handoff'])):
            raise ValueError('cultivation process binding lost its original creation game')
        game = migration['to_game']
    if migrations and (record['ownership']['game_pid'] != game['pid']
            or record['ownership']['revision'] <= migrations[-1]['from_ownership']['revision']
            or record['ledger']['state']['revision'] < migrations[-1]['handoff']['ledger_revision']):
        raise ValueError('cultivation process binding differs from the current owner')
    return game, migrations


def _replace_cultivation_budget(record, budget):
    """Copy only the existing path to the budget, retaining all history values."""
    candidate = dict(record, handoff=dict(record['handoff']))
    container = candidate['handoff']
    for _ in range(20):
        if 'automatic_cultivation' in container:
            container['automatic_cultivation'] = budget
            return candidate
        container['nonbattle_context'] = dict(container['nonbattle_context'])
        container = container['nonbattle_context']
    raise ValueError('cultivation process binding lost its original budget')


def _bind_completed_frontier(path, *, task, config, frontier, clock):
    """Consume the Native loader's verified result, never reconstruct its rules."""
    from live_decision.hif_native.native_handoff_resume import _binding

    from .ownership import write_hif_record, locked_hif_record, validate_ownership
    from .checkpoint import restore_transition
    from .run_budget import validate_budget
    from .cultivation import _budget_container

    if not isinstance(frontier, tuple) or len(frontier) != 4:
        raise ValueError('cultivation process resume needs the verified completed frontier')
    bundle, policy, row, proof = frontier
    with locked_hif_record(path):
        current = task.current()
        original = Path(path).read_bytes()
        record = json.loads(original)
        owner = validate_ownership(record['ownership'])
        gate = restore_transition(record['transition'], clock_epoch=clock.epoch)
        start = start_record(record.get('handoff'))
        if (start is None or not isinstance(record.get('run_id'), str)
                or re.fullmatch(r'hif-[a-f0-9]{32}', record['run_id']) is None
                or record.get('schema_version') != 3 or start.get('schema_version') != 1 or start.get('preset') != PRESET_ID
                or start.get('status') != 'complete' or start.get('run_id') != record['run_id']
                or start.get('ap_cost') != 20 or start.get('maximum_cultivations') != 1
                or (owner['owner'], owner['phase']) != ('task275', 'battle')
                or gate.pending is not None or gate.stopped or record['ledger'].get('pending') is not None):
            raise ValueError('cultivation process resume cannot replace an unresolved start or transaction')
        retained_profile(record['handoff'], selection=config.battle_policy)
        if start.get('additional_start') is not None:
            from .new_run import validate_additional_start
            validate_additional_start(start['additional_start'], run_id=record['run_id'],
                previous_run_id=start['previous_run_id'], status='consumed')
        game, migrations = _factory_bound_game(record, now=clock.now())
        handoff = record['handoff']
        live = handoff.get('live_context')
        if (not isinstance(row, dict) or not isinstance(proof, dict) or policy is None
                or not isinstance(live, dict) or proof.get('input_authorized') is not False
                or proof.get('computation', {}).get('restored') is not True
                or (getattr(bundle, 'run_id', None), getattr(bundle, 'revision', None)) != (live['run_id'], live['revision'])
                or not isinstance(row.get('id'), str) or re.fullmatch(r'[0-9]+-[a-z0-9-]+', row['id']) is None
                or row.get('live_return') is not True):
            raise ValueError('cultivation process resume lost its verified logical frontier')
        _binding(row, live, dict(handoff, route_run_id=record['run_id']))
        state = record['ledger'].get('state') or {}
        if state.get('run_id') != record['run_id'] or state.get('revision') != handoff['ledger_revision']:
            raise ValueError('cultivation process resume ledger changed from its published handoff')
        observed_game = row.get('endpoints', {}).get('game', {})
        if (any(key in observed_game and observed_game[key] != value for key, value in current['game'].items())
                or observed_game.get('pid') != current['game']['pid']
                or observed_game.get('integrity_rid') != 0x3000):
            raise ValueError('cultivation process resume differs from the current native game instance')
        if any(key not in observed_game for key in current['game']):
            from live_decision.hif_native.native_pending_frontier import bind_current_game_instance

            verified = bind_current_game_instance(frontier, current)[3]['current_game_instance']
            if proof.get('current_game_instance') != verified:
                raise ValueError('cultivation process resume lacks its current instance continuity proof')
        # The producer's action kind is a str Enum. Persist the small receipt
        # in its JSON form so the in-call plan and its later disk read agree;
        # do not relax the private exact-type comparator or encode old budgets.
        recovery = json.loads(json.dumps(proof.get('process_recovery'), ensure_ascii=False))
        scope = {key: deepcopy(handoff[key]) for key in ('handoff_id', 'ledger_revision', 'policy_profile', 'live_context')}
        budget = (_budget_container(handoff) or {}).get('automatic_cultivation')
        validate_budget(budget, run_id=record['run_id'], game_pid=owner['game_pid'], epoch=clock.epoch,
            now=clock.now(), max_windows=64, max_elapsed_ms=21_600_000)
        already = migrations and migrations[-1]['source'] == row['id'] and game == current['game']
        old = migrations[-1]['from_game'] if already else game
        if (not isinstance(recovery, dict) or recovery.get('old_pid') != old['pid']
                or recovery.get('new_pid') != current['game']['pid']
                or recovery.get('source') != row.get('frontier_binding_source')
                or recovery.get('logical_context') != {key: value for key, value in live.items() if key != 'round_number'}
                or recovery.get('current_context') != row.get('context_recovery', {}).get('context')
                or recovery.get('zero_new_business_inputs') is not True):
            raise ValueError('cultivation process resume differs from the verified process proof')
        if already:
            if (migrations[-1]['handoff'] != scope or migrations[-1]['process_recovery'] != recovery
                    or owner != dict(migrations[-1]['from_ownership'], game_pid=current['game']['pid'],
                        revision=migrations[-1]['from_ownership']['revision']+1)):
                raise ValueError('cultivation process resume source changed after publication')
            candidate = record
        else:
            if owner['game_pid'] != old['pid'] or current['game']['pid'] == old['pid']:
                raise ValueError('cultivation process resume cannot replace its retained game instance')
            migration = dict(schema_version=1, revision=len(migrations)+1, run_id=record['run_id'], source=row['id'],
                from_game=deepcopy(old), to_game=deepcopy(current['game']), from_ownership=deepcopy(owner),
                current_task=deepcopy(current), after_window_revision=len(budget.get('execution_windows', ())),
                published_at_ms=clock.now(), handoff=scope, process_recovery=deepcopy(recovery))
            updated = dict(budget, game_pid=current['game']['pid'], process_migrations=[*migrations, migration])
            candidate = _replace_cultivation_budget(record, updated)
            candidate['ownership'] = dict(owner, game_pid=current['game']['pid'], revision=owner['revision']+1)
            _factory_bound_game(candidate, now=clock.now())
            validate_budget(updated, run_id=record['run_id'], game_pid=current['game']['pid'], epoch=clock.epoch,
                now=clock.now(), max_windows=64, max_elapsed_ms=21_600_000)
        if task.current() != current or Path(path).read_bytes() != original:
            raise ValueError('cultivation process resume native task or source changed before publication')
        if not already:
            write_hif_record(path, candidate)
        return CultivationBindingPlan(record['run_id'], False, 0, deepcopy(start['creation_task']), current, False, False,
            tuple(deepcopy((_budget_container(candidate['handoff']) or {})['automatic_cultivation']['process_migrations'])))


def bind_cultivation_task(path, *, task, config, terminal_report, clock, additional_start=None, completed_frontier=None):
    """Allocate once for a trusted task or preserve the original live run.

    A changed SDK task never turns an active start into another AP allowance.
    The normal resume host must independently prove the current save before
    accepting the returned cross-task binding. This function grants no input.
    """
    if type(task) is not NativeCultivationTask:
        raise ValueError('cultivation factory requires its native task binding')
    config = HifConfig.from_mapping(vars(config))
    if config.mode != 'advance' or config.preset != PRESET_ID:
        raise ValueError('cultivation factory requires the supported advance preset')
    if completed_frontier is not None:
        if additional_start is not None:
            raise ValueError('cultivation process resume cannot allocate another start')
        return _bind_completed_frontier(path, task=task, config=config, frontier=completed_frontier, clock=clock)
    current = task.current()
    path = Path(path)
    try:
        raw = json.loads(path.read_bytes())
    except FileNotFoundError:
        from .ownership import locked_hif_record, write_hif_record
        from .battle_policy import selected_profile
        from .customization_runtime import configured_selection
        from .new_run import _start_candidate
        if task.entry != 'Produce' or additional_start is not None or terminal_report is not None:
            raise
        if config.battle_policy is None:
            raise ValueError('HIF new cultivation requires an explicit high-score or fast choice')
        with locked_hif_record(path):
            if path.exists():
                raw = json.loads(path.read_bytes())
            else:
                run_id = 'hif-' + uuid.uuid4().hex
                candidate = _start_candidate(run_id=run_id, game_pid=current['game']['pid'], epoch=clock.epoch,
                    profile=selected_profile(config.battle_policy), creation_task=current,
                    customization_selection=configured_selection(config), start_origin='native_produce')
                if task.current() != current:
                    raise ValueError('native Produce task changed before first cultivation')
                write_hif_record(path, candidate, create=True)
                return CultivationBindingPlan(run_id, True, 20, deepcopy(current), current, False, False)
    old_start = start_record(raw.get('handoff'))
    extra = (additional_start is not None and raw.get('run_id') == additional_start.get('previous_run_id'))
    if old_start is not None and not extra:
        if (old_start.get('run_id') != raw.get('run_id')
                or old_start.get('ap_cost') != 20 or old_start.get('maximum_cultivations') != 1
                or old_start.get('status') not in ('available', 'reserved', 'created_ap_unverified', 'complete')):
            raise ValueError('cultivation factory cannot adopt an unknown start state')
        retained_profile(raw['handoff'], selection=config.battle_policy)
        creation = old_start.get('creation_task')
        if creation is not None:
            validate_creation_task(creation, game_pid=old_start['game_pid'])
            game, migrations = _factory_bound_game(raw, now=clock.now())
            if migrations and game == current['game']:
                return CultivationBindingPlan(raw['run_id'], False, 0, deepcopy(creation), current, False, False,
                    tuple(deepcopy(migrations)))
        changed = creation != current
        if (changed and creation is not None and creation['game'] == current['game']
                and raw.get('ownership', {}).get('owner') == 'task600'):
            # A native task change does not change this game instance or its
            # successful owned route receipt. The normal host validates the
            # ledger and reads the actual successor before any new input.
            from types import SimpleNamespace

            from .checkpoint import restore_transition
            from .input_receipt import original_input_report, cancelled_unissued_report
            gate = restore_transition(raw['transition'], clock_epoch=clock.epoch)
            if (old_start['status'] == 'available' and raw['ledger'] is None
                    and raw['ownership']['phase'] == 'recovery' and not gate.stopped and gate.pending is None
                    and not any(key in old_start for key in ('request', 'ap_receipt'))
                    and 'route_input_receipt' not in raw['handoff'].get('nonbattle_context', {})):
                # A failed first observation has no save or submitted action
                # to prove. Retain the one allocation and original creation;
                # the normal producer still reads the current page before input.
                return CultivationBindingPlan(raw['run_id'], False, 0, deepcopy(creation), current, False, False)
            if (raw['ownership']['phase'] == 'recovery' and not gate.stopped and gate.pending is None
                    and raw['ledger'] is not None and raw['ledger']['pending'] is None
                    and raw['handoff'].get('terminal_source') is not None):
                from .terminal_source import bind_terminal_return
                bound, terminal = bind_terminal_return(raw, task.runtime)
                origin = bound['handoff']['terminal_return_origin']['record']
                if terminal is None or raw['ledger'] != origin['ledger']:
                    raise ValueError('terminal task resume requires the unchanged original handback ledger')
                # This is the already published battle-to-route handback.
                # The normal runtime still reads the actual result/successor;
                # no source, ledger, task budget or AP receipt is rewritten.
                return CultivationBindingPlan(raw['run_id'], False, 0, deepcopy(creation), current, False, False)
            carrier = raw['handoff'].get('nonbattle_context', {}).get('route_input_receipt')
            if gate.pending is None and isinstance(carrier, dict):
                if carrier.get('dispatch_state') == 'cancelled_unissued':
                    if gate.stopped or (raw.get('ledger') or {}).get('pending') is not None:
                        raise ValueError('unissued route cancellation still has an unresolved transaction')
                    original = restore_transition(carrier['original_transition'], clock_epoch=clock.epoch)
                    cancelled_unissued_report(SimpleNamespace(handoff=raw['handoff'], run_id=raw['run_id'],
                        game_pid=current['game']['pid'], clock=clock), original)
                    return CultivationBindingPlan(raw['run_id'], False, 0, deepcopy(creation), current, False, False)
                gate = restore_transition(carrier['original_transition'], clock_epoch=clock.epoch)
            if gate.pending is not None:
                original_input_report(SimpleNamespace(handoff=raw['handoff'], run_id=raw['run_id'],
                    game_pid=current['game']['pid'], clock=clock), gate)
                return CultivationBindingPlan(raw['run_id'], False, 0, deepcopy(creation), current, False, False)
        if (changed and creation is not None and creation['game'] == current['game']
                and raw.get('ownership', {}).get('owner') == 'task275'):
            from live_decision.hif_battle_entry import HifBattleRequest
            from live_decision.hif_live_recovery import load_automatic_chain
            from live_decision.hif_recovery_reports import matching_recovery_reports

            request = HifBattleRequest.from_record(raw, attempt_id=f'{current["task_id"]}-factory-battle',
                client_size=tuple(task.endpoint()['game']['observed_client_size']))
            # Task rebinding is not battle recovery. The existing Native
            # consumer still validates the actual pending/cancel/opening and
            # spends its original recovery budget before any input. Keep its
            # unique handoff chain; do not duplicate reason/attempt filters.
            # A published handoff can be entered by a new Maa task before the
            # first Native observation exists. Its unchanged game and complete
            # handoff are already bound above; Native still guards first entry.
            if request.live_run_id is not None or matching_recovery_reports(task.runtime, request.handoff_id):
                load_automatic_chain(task.runtime, dict(route_run_id=request.run_id,
                    handoff_id=request.handoff_id, ownership_revision=request.ownership_revision,
                    game_pid=request.game_pid, ledger_revision=request.ledger_revision,
                    policy_profile=request.policy_profile))
            return CultivationBindingPlan(raw['run_id'], False, 0, deepcopy(creation), current, False, False)
        resumed = (raw.get('handoff') or {}).get('nonbattle_context', {}).get('native_task_resume')
        verified = _verified_task_receipt(raw, resumed, current, creation)
        pending = old_start['status'] in ('reserved', 'created_ap_unverified')
        return CultivationBindingPlan(raw['run_id'], False, 0, deepcopy(creation), current,
            changed and not verified, pending or changed and not verified)
    from .new_run import authorised_interrupted_start
    interrupted = authorised_interrupted_start(raw, additional_start)
    if not interrupted and raw.get('ownership', {}).get('game_pid') != current['game']['pid']:
        raise ValueError('cultivation factory needs the current completed game instance')
    if config.battle_policy is None:
        raise ValueError('HIF new cultivation requires an explicit high-score or fast choice')
    from .customization_runtime import restore_selection, configured_selection
    selection = (restore_selection(old_start['customization_selection'], catalog=None)
        if interrupted and isinstance(old_start, dict) and 'customization_selection' in old_start
        else configured_selection(config))
    # The existing helper verifies original memory/terminal/home evidence and
    # archives the complete previous record under its own shared byte lock.
    run_id = additional_start['run_id'] if extra else 'hif-' + uuid.uuid4().hex
    def authorise():
        return task.current() == current
    state = begin_authorised_new_run(path, run_id=run_id, previous_run_id=raw['run_id'],
        game_pid=current['game']['pid'], terminal_report=terminal_report,
        authorise_new_run=authorise, clock=clock, battle_policy=config.battle_policy,
        creation_task=current, additional_start=additional_start if extra else None,
        customization_selection=selection)
    return CultivationBindingPlan(run_id, True, 20, deepcopy(state['creation_task']), current, False, False)


def confirm_cultivation_task_resume(path, *, task, config, proof, clock):
    """Accept a complete current nonbattle save without another AP allowance.

    This does not settle unknown submissions or rebind a different game. Those
    use their original transaction/process producers before this acknowledgement.
    """
    from .hif import active_hif_role
    from .ledger import HifLedgerState
    from .ownership import write_hif_record, locked_hif_record, validate_ownership
    from .checkpoint import restore_transition
    from .run_budget import validate_budget
    from .cultivation import _budget_container
    if type(task) is not NativeCultivationTask or not isinstance(proof, SameSaveProof):
        raise ValueError('cultivation task resume requires its native task and full current proof')
    config = HifConfig.from_mapping(vars(config))
    if config.mode != 'advance':
        raise ValueError('cultivation task resume cannot publish from observe mode')
    current = task.current()
    path = Path(path)
    with locked_hif_record(path):
        raw = json.loads(path.read_bytes())
        owner = validate_ownership(raw['ownership'])
        gate = restore_transition(raw['transition'], clock_epoch=clock.epoch)
        start = start_record(raw.get('handoff'))
        ledger = raw.get('ledger') or {}
        state = ledger.get('state') or {}
        if (start is None or start.get('run_id') != raw['run_id'] or start.get('status') != 'complete'
                or start.get('ap_cost') != 20 or start.get('maximum_cultivations') != 1
                or (owner['owner'], owner['phase']) != ('task600', 'nonbattle')
                or owner['game_pid'] != current['game']['pid'] or start['game_pid'] != current['game']['pid']
                or ledger.get('pending') is not None or gate.pending is not None or gate.stopped
                or not state or not isinstance(proof.state, HifLedgerState)
                or proof.run_id != raw['run_id'] or proof.game_pid != owner['game_pid']
                or proof.identity != (*active_hif_role(raw), 'HIF')
                or not isinstance(proof.probe_id, str) or not proof.probe_id
                or type(proof.started_at_ms) is not int or type(proof.observed_at_ms) is not int
                or not 0 <= proof.started_at_ms <= proof.observed_at_ms <= clock.now()
                or _semantic(asdict(proof.state)) != _semantic(state)):
            raise ValueError('cultivation task resume cannot replace an unresolved or different save')
        retained_profile(raw['handoff'], selection=config.battle_policy)
        validate_creation_task(start.get('creation_task'), game_pid=start['game_pid'])
        batches = (proof.identity_frame_ids, proof.inventory_frame_ids, proof.resource_frame_ids)
        for index, frames in enumerate(batches):
            _frames(frames, game_pid=proof.game_pid, started=proof.started_at_ms,
                observed=proof.observed_at_ms, count=None if index == 1 else 1)
        flat = [key for frames in batches for key in frames]
        if len(set(flat)) != len(flat):
            raise ValueError('cultivation task resume reused identity, inventory or resource sources')
        _fresh_resource_batch(proof, now_ms=clock.now())
        container = _budget_container(raw['handoff']) or {}
        validate_budget(container.get('automatic_cultivation'), run_id=raw['run_id'], game_pid=owner['game_pid'],
            epoch=clock.epoch, now=clock.now(), max_windows=64, max_elapsed_ms=21_600_000)
        receipt = dict(schema_version=1, run_id=raw['run_id'], creation_task=deepcopy(start['creation_task']),
            current_task=deepcopy(current), source_revision=state['revision'], proof=asdict(proof))
        handoff = deepcopy(raw['handoff'])
        previous = handoff.get('nonbattle_context', {}).get('native_task_resume')
        if previous == json.loads(json.dumps(receipt)):
            return CultivationBindingPlan(raw['run_id'], False, 0, deepcopy(start['creation_task']), current, False, False)
        if previous is not None:
            receipt['previous'] = deepcopy(previous)
        handoff.setdefault('nonbattle_context', {})['native_task_resume'] = receipt
        if task.current() != current or json.loads(path.read_bytes()) != raw:
            raise ValueError('cultivation task resume source changed before publication')
        write_hif_record(path, dict(raw, handoff=handoff))
    return CultivationBindingPlan(raw['run_id'], False, 0, deepcopy(start['creation_task']), current, False, False)


def _same_private_json(left, right):
    """Compare one stable JSON pair without expanding shared subgraphs again."""
    compared = {}

    def same(left, right):
        if left is right:
            return True
        dictionaries = isinstance(left, dict) and isinstance(right, dict)
        lists = isinstance(left, list) and isinstance(right, list)
        if not dictionaries and not lists:
            # Python equality alone treats true, 1 and 1.0 as equal.
            return type(left) is type(right) and left == right
        pair = (id(left), id(right))
        if pair not in compared:
            if dictionaries:
                equal = (left.keys() == right.keys()
                    and all(same(child, right[key]) for key, child in left.items()))
            else:
                equal = len(left) == len(right) and all(same(a, b) for a, b in zip(left, right, strict=True))
            compared[pair] = equal
        return compared[pair]

    return same(left, right)


def _private_control_interner():
    from live_decision.hif_live_io import freeze_record_history

    controls = {}

    def intern(value):
        key = tuple(value.get(field) for field in (
            'root', 'previous_record', 'sequence', 'last_checked_at', 'active_parent_deadline'))
        try:
            bucket = controls.setdefault(key, [])
        except TypeError:
            return value
        for saved in bucket:
            if _same_private_json(value, saved):
                return saved
        frozen = freeze_record_history(value)
        bucket.append(frozen)
        return frozen

    return intern


def _decode_private_control_history(raw):
    """Intern exact old prefixes while JSON decodes, avoiding a second walk.

    The decoder still reads every byte and constructs every JSON field. Only
    exactly equal source_control objects are shared, with the same immutable
    representation as the existing post-decode path.
    """
    intern = _private_control_interner()

    def decoded(value):
        child = value.get('source_control')
        if isinstance(child, dict):
            value['source_control'] = intern(child)
        return value

    return json.loads(raw, object_hook=decoded)


def _share_private_control_history(record):
    """Share exact immutable source controls in an already decoded record."""
    intern = _private_control_interner()

    def visit(value, *, source=False):
        if isinstance(value, list):
            for index, child in enumerate(value):
                value[index] = visit(child)
        elif isinstance(value, dict):
            for key, child in value.items():
                value[key] = visit(child, source=key == 'source_control')
            if source:
                return intern(value)
        return value

    return visit(record)


class CultivationRecordBinding:
    """Read the factory's retained run, independently of the legacy test run ID."""
    def __init__(self, task, plan, *, path, clock):
        if (type(task) is not NativeCultivationTask or not isinstance(plan, CultivationBindingPlan)
                or plan.observe_only or re.fullmatch(r'hif-[a-f0-9]{32}', plan.run_id) is None
                or task.current() != plan.current_task):
            raise ValueError('cultivation factory record is not bound to its current native task')
        self.task, self.plan, self.path, self.clock = task, plan, Path(path), clock
        self.run_id = plan.run_id
        self._record_decode = {}
        value = self.read(self.path)
        start = start_record(value['handoff'])
        from .new_run import initial_native_start
        if initial_native_start(start):
            return
        disposition = start.get('previous_disposition')
        if disposition not in (None, 'interrupted_external_unverified'):
            raise ValueError('cultivation factory previous-run disposition is unknown')
        previous = self.path.parent/('interrupted' if disposition else 'completed')/(start['previous_run_id']+'.json')
        if (Path(start['previous_record']).resolve() != previous.resolve() or not previous.is_file()
                or json.loads(previous.read_bytes()).get('run_id') != start['previous_run_id']):
            raise ValueError('cultivation factory lost its original completed archive')

    def read(self, path, *, read_sink=None, pixel_decode=None):
        from .ownership import validate_ownership
        from .run_budget import validate_budget
        from .cultivation import _budget_container
        if Path(path) != self.path:
            raise ValueError('cultivation factory storage path changed')
        raw = self.path.read_bytes()
        if pixel_decode is not None and read_sink is None and pixel_decode.get('raw') == raw:
            record = pixel_decode['record']
        else:
            # Reuse parsed data only after reading and comparing all current
            # bytes. Never expose this private copy or retain a validation
            # result: the ordinary owner/budget checks below always run.
            decoded = self._record_decode
            if decoded.get('raw') != raw:
                private = _decode_private_control_history(raw)
                decoded.clear()
                decoded.update(raw=raw, record=private)
            record = deepcopy(decoded['record'])
            if pixel_decode is not None and read_sink is None:
                pixel_decode.clear()
                pixel_decode.update(raw=raw, record=record)
        if (not isinstance(record, dict) or set(record) != {'schema_version', 'run_id', 'ownership', 'transition', 'ledger', 'handoff'}
                or record['schema_version'] != 3 or record['run_id'] != self.run_id):
            raise ValueError('cultivation factory current run changed')
        owner = validate_ownership(record['ownership'])
        start = start_record(record['handoff'])
        game, migrations = _factory_bound_game(record, now=self.clock.now())
        if not _same_private_json(migrations, list(self.plan.process_migrations)):
            raise ValueError('cultivation factory process binding changed during its native task')
        if (not isinstance(start, dict) or start.get('schema_version') != 1 or start.get('run_id') != self.run_id
                or start.get('creation_task') != self.plan.creation_task
                or start.get('preset') != PRESET_ID or start.get('ap_cost') != 20 or start.get('maximum_cultivations') != 1
                or start.get('status') not in ('available', 'reserved', 'created_ap_unverified', 'complete')
                or game != self.plan.current_task['game']
                or owner['game_pid'] != self.plan.current_task['game']['pid']):
            raise ValueError('cultivation factory changed its original task, game or AP budget')
        if start.get('additional_start') is not None:
            from .new_run import validate_additional_start
            validate_additional_start(start['additional_start'], run_id=self.run_id,
                previous_run_id=start['previous_run_id'], status={
                    'available': 'available', 'reserved': 'pending',
                    'created_ap_unverified': 'consumed', 'complete': 'consumed'}[start['status']])
        validate_creation_task(start['creation_task'], game_pid=start['game_pid'])
        retained_profile(record['handoff'])
        if record['ledger'] is None:
            if ((owner['owner'], owner['phase']) != ('task600', 'recovery')
                    or (record['handoff'] or {}).get('live_context') is not None
                    or (record['handoff'] or {}).get('terminal_receipt') is not None):
                raise ValueError('cultivation factory has no complete current nonbattle ledger')
        elif record['ledger'].get('state', {}).get('run_id') != self.run_id:
            raise ValueError('cultivation factory ledger belongs to another run')
        container = _budget_container(record['handoff']) or {}
        budget = container.get('automatic_cultivation')
        if budget is not None:
            validate_budget(budget, run_id=self.run_id, game_pid=owner['game_pid'], epoch=self.clock.epoch,
                now=self.clock.now(), max_windows=64, max_elapsed_ms=21_600_000)
        elif record['ledger'] is not None:
            raise ValueError('cultivation factory cannot reconstruct a lost cumulative budget')
        if read_sink is not None:
            read_sink(raw, record)
        return record

    def close(self):
        self._record_decode.clear()
        self.task.close()
