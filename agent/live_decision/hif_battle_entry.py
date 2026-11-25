"""Same-process battle call for the existing Maa cultivation scheduler.

The caller releases its route lease, snapshots the published handoff, then
calls this function on the same original Maa context. No launcher, command
file, game creation or implicit input ownership transfer is performed here.
"""
from __future__ import annotations

import re
import csv
import json
import time
from copy import deepcopy
from pathlib import Path
from contextlib import nullcontext
from dataclasses import asdict, dataclass

from route.ownership import HifInputSession, valid_terminal_frame_ids
from route.battle_policy import FORMAL_PROFILES

from .hif_session import HifRoundContext
from .hif_live_host import HifLiveHost, HifBattleContext
from .hif_capture_adapter import HifCaptureAdapter, _TaskerProxy, _existing_adapter
from .hif_navigation_recovery import load_battle_navigation_source as load_battle_navigation_source


@dataclass(frozen=True)
class HifBattleRequest:
    attempt_id: str
    run_id: str
    handoff_id: str
    ownership_revision: int
    ledger_revision: int | None
    round_number: int
    policy_profile: str
    game_pid: int
    client_size: tuple[int, int]
    live_run_id: str | None
    live_revision: int | None

    def __post_init__(self):
        if (not isinstance(self.attempt_id, str) or re.fullmatch(r'[0-9]+-[a-z0-9-]+', self.attempt_id) is None
                or any(not isinstance(value, str) or not value.strip() for value in (self.run_id, self.handoff_id))
                or any(type(value) is not int or value < 1 for value in (self.ownership_revision, self.game_pid))
                or not self._valid_ledger_revision()
                or type(self.round_number) is not int or self.round_number not in (1, 2)
                or self.policy_profile not in FORMAL_PROFILES
                or not isinstance(self.client_size, tuple) or len(self.client_size) != 2
                or any(type(value) is not int or value <= 0 for value in self.client_size)
                or ((self.live_run_id is None) != (self.live_revision is None))
                or self.live_run_id is not None and (
                    not isinstance(self.live_run_id, str) or not self.live_run_id.strip()
                    or type(self.live_revision) is not int or self.live_revision < 1)):
            raise ValueError('HIF battle request identity, revision, round or mode is invalid')

    def _valid_ledger_revision(self):
        return type(self.ledger_revision) is int and self.ledger_revision >= 0

    @classmethod
    def from_record(cls, record, *, attempt_id, client_size):
        request = cls.from_identity(record, attempt_id=attempt_id, client_size=client_size)
        request.verify_record(record)
        return request

    @classmethod
    def from_identity(cls, record, *, attempt_id, client_size):
        """Read-only identity for navigation recovery, never battle admission."""
        owner, handoff = record['ownership'], record['handoff']
        current = handoff.get('source') == 'current_battle_page'
        if (owner['owner'] != 'task275' or owner['phase'] != 'battle'
                or not current and handoff.get('to_owner') != 'task275'):
            raise ValueError('HIF battle request needs the already published battle handoff')
        live = handoff.get('live_context') or {}
        kind = CurrentPageBattleRequest if current else cls
        request = kind(attempt_id, record['run_id'], handoff['handoff_id'], owner['revision'],
            handoff['ledger_revision'], handoff['round_number'], handoff['policy_profile'],
            owner['game_pid'], tuple(client_size), live.get('run_id'), live.get('revision'))
        request.verify_identity(record)
        return request

    def verify_identity(self, record):
        owner, handoff = record['ownership'], record['handoff']
        live = handoff.get('live_context') or {}
        if (record['run_id'] != self.run_id
                or owner != dict(owner='task275', phase='battle', revision=self.ownership_revision, game_pid=self.game_pid)
                or not self._matches_origin(handoff) or handoff.get('handoff_id') != self.handoff_id
                or handoff.get('ledger_revision') != self.ledger_revision
                or handoff.get('round_number') != self.round_number
                or handoff.get('policy_profile') != self.policy_profile
                or (live.get('run_id'), live.get('revision')) != (self.live_run_id, self.live_revision)
                or live and live.get('round_number') != self.round_number
                or not self._matches_ledger(record['ledger'])):
            raise ValueError('HIF battle request is stale or differs from the active handoff')

    def _matches_ledger(self, ledger):
        return ledger['state']['revision'] == self.ledger_revision and ledger['pending'] is None

    def _matches_origin(self, handoff):
        return handoff.get('to_owner') == 'task275'

    def verify_record(self, record):
        self.verify_identity(record)
        if record['transition']['pending'] is not None or record['transition']['stopped'] is not False:
            raise ValueError('HIF battle request is stale or differs from the active handoff')


class CurrentPageBattleRequest(HifBattleRequest):
    """A battle witnessed on screen has no route inventory revision."""

    def _valid_ledger_revision(self):
        return self.ledger_revision is None

    def _matches_ledger(self, ledger):
        return ledger is None

    def _matches_origin(self, handoff):
        return handoff.get('source') == 'current_battle_page'

    def verify_record(self, record):
        self.verify_identity(record)


def run_current_page_battle(context, *, path, runtime, run_id, game_pid, client_size,
        policy_profile, authorise_task, expected_round=None):
    """Use the visible opening or this battle's original reports on recovery."""
    from .hif_entry_round import current_opening_round
    from .hif_capture_adapter import adapt_hif_context
    from .hif_current_battle_session import CurrentBattleSession, current_battle_round

    number = current_battle_round(runtime, run_id, game_pid)
    if not authorise_task():
        raise ValueError('current battle task is no longer active')
    if number is None:
        view = adapt_hif_context(context)
        controller = view.tasker.controller
        if not controller.post_screencap().wait().succeeded:
            raise ValueError('current battle capture failed')
        number = current_opening_round(view, controller.cached_image)
        if number is None:
            raise ValueError('current page is not a playable HIF opening')
    if expected_round is not None and number != expected_round:
        raise ValueError('current HIF battle does not match the cultivation round')

    # The supported HIF preset's plan remains fixed. All cards and bottles come
    # from the existing live opening observer, including after manual cultivation.
    with CurrentBattleSession(path, runtime=runtime, run_id=run_id, game_pid=game_pid,
            round_number=number, policy_profile=policy_profile, plan='sense', authorise_task=authorise_task) as session:
        if session.completed:
            raise ValueError('current battle already has a terminal receipt')
        attempt_id = f'{time.time_ns() // 1_000_000}-{run_id}-current-battle'
        request = HifBattleRequest.from_record(session.record, attempt_id=attempt_id, client_size=client_size)
        return run_hif_battle(context, path=path, request=request, runtime=runtime,
            authorised_run_id=run_id, current_session=session)


def _battle_context(context, client_size):
    """Do not attach a battle lease to the scheduler's shared capture adapter."""
    battle = HifBattleContext(context)
    adapter = _existing_adapter(context.tasker.controller)
    if adapter is not None:
        if adapter._hif_input_guard is not None:
            raise ValueError('HIF caller context already has an input owner; use its original Maa context')
        # A fresh view over the same Maa endpoint preserves neither captured
        # frames nor an old input guard. Never unwrap an ownership-bound view.
        controller = HifCaptureAdapter(adapter._hif_wrapped)
        object.__setattr__(battle, '_hif_battle_tasker', _TaskerProxy(context.tasker, controller))
    battle.gkh_hif_client_size = client_size
    return battle


def _parameters(handoff):
    return dict(live_decision_mode='opening_execute', hif_policy_profile=handoff['policy_profile'],
        expected_plan=handoff['plan'], provider_timeout_ms=500, max_snapshot_age_ms=1000,
        minimum_identity_confidence=.85, debug_identity_threshold=.70, debug_identity_min_margin=.05,
        debug_displayed_effective_cost_min_confidence=.85, debug_preview_delta_min_confidence=.85,
        debug_resource_absolute_min_confidence=.85, debug_expected_initial_hand_count=3,
        debug_require_foreground_window=False)


def _boundary(report):
    if report is None:
        return dict(kind='not_inspected', record=None, transaction_closed=None)
    events = report.get('live_events', ())
    if report.get('transaction_closed') is True:
        kind = 'terminal' if report.get('terminal_settlement') else 'settled'
    elif report.get('native_pending_source') or any(
            event.get('execution_submit_count', 0) or event.get('end_turn_submit_count', 0) for event in events):
        kind = 'submitted_pending'
    elif report.get('gray_selection_journal') or report.get('native_selected_source') or any(
            event.get('execution_selection_click_count') == 1 for event in events):
        kind = 'selected_unsubmitted'
    else:
        kind = 'unresolved'
    return dict(kind=kind, record=report.get('id'), transaction_closed=report.get('transaction_closed') is True)


def verify_prepared_terminal_return(request, shared_record, report):
    """Prove publication from the current shared owner, without rewriting a report.

    A prepared report does not itself assert a successful handback. Only its
    exact source/receipt in the atomically published recovery envelope does.
    This proof grants no input and does not refresh any execution budget.
    """
    if request is not None and not isinstance(request, HifBattleRequest):
        raise ValueError('HIF terminal recovery request is invalid')
    if not isinstance(shared_record, dict) or not isinstance(report, dict):
        raise ValueError('HIF terminal publication requires complete records')
    if (any(not isinstance(shared_record.get(key), dict) for key in ('ownership', 'handoff', 'ledger', 'transition'))
            or any(not isinstance(report.get(key), dict) for key in
                ('handoff_opening', 'terminal_settlement', 'endpoints', 'formal_params', 'round_context'))
            or not isinstance(shared_record['handoff'].get('live_context'), dict)
            or not isinstance(shared_record['ledger'].get('state'), dict)
            or not isinstance(report['endpoints'].get('game'), dict)):
        raise ValueError('HIF terminal publication contains incomplete identities')
    owner = shared_record.get('ownership') or {}
    handoff = shared_record.get('handoff') or {}
    live = handoff.get('live_context') or {}
    opening = report.get('handoff_opening') or {}
    terminal = report.get('terminal_settlement') or {}
    ledger = shared_record.get('ledger') or {}
    before_revision = report.get('handoff_ownership_revision')
    source = handoff.get('terminal_source')
    if (shared_record.get('schema_version') != 3
            or not isinstance(source, str) or re.fullmatch(r'[0-9]+-[a-z0-9-]+', source) is None
            or report.get('id') != source or report.get('handoff_return_prepared') is not True
            or type(before_revision) is not int or before_revision < 1
            or owner != dict(owner='task600', phase='recovery', revision=before_revision + 1,
                game_pid=report.get('endpoints', {}).get('game', {}).get('pid'))
            or type(owner.get('game_pid')) is not int or owner['game_pid'] < 1
            or not isinstance(shared_record.get('run_id'), str) or not shared_record['run_id']
            or handoff.get('to_owner') != 'task275' or type(handoff.get('round_number')) is not int
            or handoff['round_number'] not in (1, 2) or handoff.get('policy_profile') not in FORMAL_PROFILES
            or type(handoff.get('ledger_revision')) is not int or handoff['ledger_revision'] < 0
            or report.get('status') != 'TERMINAL_VERIFIED' or report.get('live_return') is not True
            or report.get('transaction_closed') is not True or report.get('error')
            or report.get('returned_to_nonbattle') is not None and type(report['returned_to_nonbattle']) is not bool
            or type(report.get('total_input_count')) is not int or report['total_input_count'] < 0
            or report.get('handoff_id') != handoff.get('handoff_id')
            or not isinstance(handoff.get('handoff_id'), str) or not handoff['handoff_id']
            or not isinstance(live, dict) or not live
            or (opening.get('route_run_id'), opening.get('handoff_id'), opening.get('ledger_revision'),
                opening.get('live_run_id'), opening.get('live_revision'), opening.get('round_number'),
                opening.get('policy_profile')) != (
                    shared_record.get('run_id'), handoff.get('handoff_id'), handoff.get('ledger_revision'),
                    live.get('run_id'), live.get('revision'), handoff.get('round_number'), handoff.get('policy_profile'))
            or report.get('formal_params', {}).get('hif_policy_profile') != handoff.get('policy_profile')
            or live.get('round_number') != handoff.get('round_number')
            or ledger.get('pending') is not None
            or ledger.get('state', {}).get('revision') != handoff.get('ledger_revision')
            or ledger.get('state', {}).get('run_id') != shared_record.get('run_id')
            or shared_record.get('transition', {}).get('pending') is not None
            or shared_record.get('transition', {}).get('stopped') is not False
            or terminal != handoff.get('terminal_receipt')
            or terminal.get('accepted') is not True or terminal.get('terminal') is not True
            or (terminal.get('run_id'), terminal.get('revision'), terminal.get('round_number'))
                != (live.get('run_id'), live.get('revision'), handoff.get('round_number'))
            or type(terminal.get('score')) is not int or terminal['score'] < 0
            or not valid_terminal_frame_ids(terminal.get('frame_ids'))
            or set(terminal['frame_ids']).intersection(handoff.get('source_frame_ids', ()))):
        raise ValueError('HIF prepared terminal source differs from its published handback')
    try:
        round_context = HifRoundContext(**report['round_context'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('HIF prepared terminal notebook is incomplete') from error
    if ((round_context.run_id, round_context.revision, round_context.round_number)
            != (live['run_id'], live['revision'], handoff['round_number'])
            or set(round_context.frame_ids).intersection(terminal['frame_ids'])):
        raise ValueError('HIF prepared terminal notebook belongs to another battle')
    if request is not None and (
            (request.run_id, request.handoff_id, request.ownership_revision, request.ledger_revision,
                request.round_number, request.policy_profile, request.game_pid)
            != (shared_record['run_id'], handoff['handoff_id'], before_revision, handoff['ledger_revision'],
                handoff['round_number'], handoff['policy_profile'], owner['game_pid'])
            or request.live_run_id is not None and (request.live_run_id, request.live_revision)
                != (live['run_id'], live['revision'])):
        raise ValueError('HIF prepared terminal publication differs from the requested battle')
    return dict(kind='verified_prepared_terminal_return', terminal_source=source,
        run_id=shared_record['run_id'], handoff_id=handoff['handoff_id'], round_number=handoff['round_number'],
        ownership_revision=owner['revision'], ledger_revision=handoff['ledger_revision'],
        live_run_id=live['run_id'], live_revision=live['revision'])


def load_prepared_terminal_return(runtime, shared_record, *, request=None):
    """Load only the current published source; never discover it by file time."""
    source = (shared_record.get('handoff') or {}).get('terminal_source')
    if not isinstance(source, str) or re.fullmatch(r'[0-9]+-[a-z0-9-]+', source) is None:
        raise ValueError('HIF published terminal source is missing or invalid')
    report = json.loads((Path(runtime) / f'result-{source}.json').read_text(encoding='utf-8'))
    return report, verify_prepared_terminal_return(request, shared_record, report)


def _verify_return(request, before, after, report):
    verify_prepared_terminal_return(request, after, report)
    handoff = after['handoff']
    live, terminal = handoff.get('live_context'), report.get('terminal_settlement')
    opening = report.get('handoff_opening', {})
    if (report.get('status') != 'TERMINAL_VERIFIED' or report.get('live_return') is not True
            or report.get('transaction_closed') is not True or report.get('returned_to_nonbattle') is not True
            or after['run_id'] != request.run_id
            or after['ownership'] != dict(owner='task600', phase='recovery',
                revision=request.ownership_revision + 1, game_pid=request.game_pid)
            or after['ledger'] != before['ledger'] or after['transition'] != before['transition']
            or handoff.get('handoff_id') != request.handoff_id or handoff.get('round_number') != request.round_number
            or handoff.get('ledger_revision') != request.ledger_revision
            or handoff.get('policy_profile') != request.policy_profile
            or not isinstance(live, dict) or not isinstance(terminal, dict)
            or live.get('round_number') != request.round_number
            or request.live_run_id is not None and (live.get('run_id'), live.get('revision'))
                != (request.live_run_id, request.live_revision)
            or handoff.get('terminal_receipt') != terminal
            or terminal.get('accepted') is not True or terminal.get('terminal') is not True
            or (terminal.get('run_id'), terminal.get('revision'), terminal.get('round_number'))
                != (live.get('run_id'), live.get('revision'), request.round_number)
            or type(terminal.get('score')) is not int or terminal['score'] < 0
            or not valid_terminal_frame_ids(terminal.get('frame_ids'))
            or set(terminal['frame_ids']) & set(handoff['source_frame_ids'])
            or report.get('endpoints', {}).get('game', {}).get('pid') != request.game_pid
            or (opening.get('route_run_id'), opening.get('handoff_id'), opening.get('live_run_id'),
                opening.get('live_revision'), opening.get('round_number'))
                != (request.run_id, request.handoff_id, live.get('run_id'), live.get('revision'), request.round_number)):
        raise ValueError('HIF terminal result does not prove this battle returned to nonbattle')
    round_context = HifRoundContext(**report['round_context'])
    if (round_context.run_id, round_context.revision, round_context.round_number) != (
            live['run_id'], live['revision'], request.round_number):
        raise ValueError('HIF terminal round proof belongs to another battle')


def run_hif_battle(context, *, request, path, runtime, authorised_run_id, observer_module=None, legacy_resume_source=None,
        parent_deadline=None, identity_readback=None, current_session=None):
    """Return a structured STOP or verified handback; never relay/retry a STOP.

    ``request`` is a snapshot, not permission. ``authorised_run_id`` comes from
    the existing host's approved scope. The lease, Maa endpoints and fresh input
    checks remain mandatory. The optional legacy source is for old development
    receipts only; normal schedulers omit it and use automatic recovery.
    An automatic battle retains its original action and search budgets across
    process restarts. Historical scheduler timestamps do not end a battle.

    On STOP the native reports, pending transaction, lease owner and budgets
    are retained. ``terminal_report`` is available only after a verified return
    and can be passed directly to the existing nonbattle ending producer.
    """
    if not isinstance(request, HifBattleRequest):
        raise ValueError('HIF scheduler requires an explicit battle request')
    result = dict(schema_version=1, request=asdict(request), status='STOP', returned_to_nonbattle=False,
        handoff_id=request.handoff_id, policy_profile=request.policy_profile,
        last_record=None, records=[], input_count=0, ownership=None, boundary=_boundary(None),
        terminal_settlement=None, terminal_report=None, automatic_battle=None, automatic_recovery_stop=None,
        error=None, error_type=None, error_stage=None, stop_reason=None)
    host, report, stage = None, None, 'authorization'
    try:
        if not isinstance(authorised_run_id, str) or authorised_run_id != request.run_id:
            raise ValueError('HIF battle is outside the host authorised run')
        stage = 'budget'
        from .hif_live_recovery import validate_parent_deadline
        validate_parent_deadline(parent_deadline)
        stage = 'ownership'
        if isinstance(request, CurrentPageBattleRequest):
            from .hif_current_battle_session import CurrentBattleSession
            if (not isinstance(current_session, CurrentBattleSession) or not current_session.authorise_input()
                    or current_session.handoff_id != request.handoff_id):
                raise ValueError('current battle requires its active original Maa session')
            lease = nullcontext(current_session)
        else:
            if current_session is not None:
                raise ValueError('published battle uses its route input session')
            lease = HifInputSession(path, owner='task275', run_id=request.run_id,
                game_pid=request.game_pid, handoff_id=request.handoff_id)
        with lease as session:
            before = session.record
            result['ownership'] = before['ownership']
            request.verify_record(before)
            stage = 'binding'
            battle = _battle_context(context, request.client_size)
            if observer_module is None:
                from custom.action import live_decision as observer_module
            catalog = observer_module.SkillCardFeatureCatalog.load(observer_module.CATALOG_PATH)
            with (observer_module.CATALOG_PATH.parent / 'p_drinks.csv').open(encoding='utf-8-sig', newline='') as stream:
                drinks = {row['id'] for row in csv.DictReader(stream)}
            host = HifLiveHost(battle, session, runtime=Path(runtime), attempt_id=request.attempt_id + '-live',
                params=_parameters(before['handoff']), card_catalog=catalog, drink_ids=drinks,
                observer_module=observer_module, host_attempt=request.attempt_id)
            stage = 'battle'
            try:
                options = ({'automatic_recovery': True} if legacy_resume_source is None
                    else {'resume_source': legacy_resume_source})
                if parent_deadline is not None:
                    options['parent_deadline'] = parent_deadline
                if identity_readback is not None:
                    options['identity_readback'] = identity_readback
                report = host.run(**options)
                result.update(last_record=report['id'], records=list(report['continuation_records']),
                    input_count=report.get('total_input_count'), boundary=_boundary(report),
                    automatic_battle=deepcopy(report.get('automatic_battle')),
                    automatic_recovery_stop=report.get('automatic_recovery_stop'),
                    host_status=report['status'], error=report.get('error'), error_type=report.get('error_type'))
                if type(result['input_count']) is not int or result['input_count'] < 0:
                    result['input_count'] = None
                    raise ValueError('HIF battle input accounting is incomplete')
                if current_session is not None and report.get('current_battle_finished') is True:
                    stage = 'terminal_result'
                    handoff = session.record['handoff']
                    terminal = report.get('terminal_settlement')
                    if (not isinstance(terminal, dict) or terminal.get('accepted') is not True
                            or terminal.get('terminal') is not True or terminal.get('round_number') != request.round_number
                            or type(terminal.get('score')) is not int or terminal['score'] < 0
                            or not valid_terminal_frame_ids(terminal.get('frame_ids'))
                            or report.get('status') != 'TERMINAL_VERIFIED' or report.get('transaction_closed') is not True
                            or terminal != handoff.get('terminal_receipt') or report['id'] != handoff.get('terminal_source')):
                        raise ValueError('current battle result differs from its observed terminal receipt')
                    result.update(status='TERMINAL_VERIFIED', current_battle_finished=True,
                        terminal_settlement=deepcopy(terminal), terminal_report=deepcopy(report))
                elif report.get('returned_to_nonbattle') is True:
                    stage = 'terminal_return'
                    _verify_return(request, before, session.record, report)
                    result.update(status='TERMINAL_VERIFIED', returned_to_nonbattle=True,
                        terminal_settlement=deepcopy(report['terminal_settlement']), terminal_report=deepcopy(report))
                else:
                    result['error_stage'] = 'battle'
                    failures = [event.get('reason') for event in report.get('live_events', ())
                        if event.get('accepted') is False and event.get('reason')]
                    result['stop_reason'] = (report.get('error') or report.get('automatic_recovery_stop')
                        or report.get('continuation_stop') or (failures[-1] if failures else 'observer_stopped'))
            finally:
                result['ownership'] = session.record['ownership']
    except Exception as error:
        if host is not None and host.result is not None:
            report = host.result
            count = report.get('total_input_count')
            result.update(last_record=report.get('id'), records=list(host.records), boundary=_boundary(report),
                input_count=count if type(count) is int and count >= 0 else None,
                automatic_battle=deepcopy(report.get('automatic_battle')))
        result.update(status='STOP', returned_to_nonbattle=False, terminal_report=None,
            error=str(error), error_type=type(error).__name__, error_stage=stage,
            stop_reason=str(error) or type(error).__name__)
    return result
