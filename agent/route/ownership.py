"""One process lease and one durable input owner in the existing HIF record.

Ownership never grants capture/input permission or verifies a Windows endpoint.
Both clients must keep this lease while using their separate permission gates.
"""
from __future__ import annotations

import os
import re
import json
import tempfile
from copy import deepcopy
from pathlib import Path
from contextlib import contextmanager

from .engine import has_single_frame_id
from .checkpoint import restore_transition
from .battle_policy import FORMAL_PROFILES

OWNERS = frozenset({'task600', 'task275'})
PHASES = {'task600': {'recovery', 'nonbattle'}, 'task275': {'battle'}}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def validate_ownership(raw):
    if (not isinstance(raw, dict) or set(raw) != {'revision', 'owner', 'phase', 'game_pid'}
            or type(raw['revision']) is not int or raw['revision'] < 1
            or not isinstance(raw['owner'], str) or raw['owner'] not in OWNERS
            or not isinstance(raw['phase'], str) or raw['phase'] not in PHASES[raw['owner']]
            or type(raw['game_pid']) is not int or raw['game_pid'] <= 0):
        raise ValueError('invalid HIF input ownership')
    return dict(raw)


@contextmanager
def locked_hif_record(path):
    """The same byte lock as HifRouteHost; a competing client cannot enter."""
    if os.name != 'nt':
        raise RuntimeError('HIF input ownership requires Windows locking')
    import msvcrt
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a+b') as lease:
        lease.seek(0)
        msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            lease.seek(0)
            msvcrt.locking(lease.fileno(), msvcrt.LK_UNLCK, 1)


def write_hif_record(path, raw, *, create=False):
    """Existing atomic replacement/exclusive creation, without a second ledger."""
    path = Path(path)
    data = json.dumps(raw, ensure_ascii=False, allow_nan=False).encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent,
                prefix=path.name + '.', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if create:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_owned_record(path, *, run_id, clock_epoch):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(raw, dict) or type(raw.get('schema_version')) is not int
            or raw['schema_version'] != 3
            or set(raw) != {'schema_version', 'run_id', 'transition', 'ledger', 'ownership', 'handoff'}
            or raw['run_id'] != run_id):
        raise ValueError('HIF shared record identity/version mismatch')
    validate_ownership(raw['ownership'])
    restore_transition(raw['transition'], clock_epoch=clock_epoch)
    return raw


def valid_terminal_frame_ids(values):
    return (isinstance(values, (tuple, list)) and bool(values)
        and all(_text(value) for value in values) and len(set(values)) == len(values))


def _verify_terminal_source(record, receipt, source, runtime):
    """Read the already persisted report before publishing returned ownership."""
    if (not isinstance(source, str) or re.fullmatch(r'[0-9]+-[a-z0-9-]+', source) is None
            or not isinstance(runtime, (str, Path))):
        raise ValueError('HIF battle terminal report source is invalid')
    report = json.loads((Path(runtime).resolve() / f'result-{source}.json').read_text(encoding='utf-8-sig'))
    handoff = record['handoff']
    live, opening, round_context = handoff.get('live_context') or {}, report.get('handoff_opening') or {}, report.get('round_context') or {}
    expected = (record['run_id'], handoff['handoff_id'], live.get('run_id'), live.get('revision'), handoff['round_number'])
    if (report.get('id') != source or report.get('status') != 'TERMINAL_VERIFIED'
            or report.get('live_return') is not True or report.get('transaction_closed') is not True
            or report.get('handoff_return_prepared') is not True or report.get('terminal_settlement') != receipt
            or report.get('handoff_ownership_revision') != record['ownership']['revision']
            or report.get('endpoints', {}).get('game', {}).get('pid') != record['ownership']['game_pid']
            or tuple(opening.get(key) for key in ('route_run_id', 'handoff_id', 'live_run_id', 'live_revision', 'round_number')) != expected
            or tuple(round_context.get(key) for key in ('run_id', 'revision', 'round_number')) != expected[2:]):
        raise ValueError('HIF persisted terminal report differs from this battle return')


class HifInputSession:
    """Hold the shared lock throughout one client's bounded Maa invocation.

    ``record`` is a detached snapshot. It cannot mutate the active lease.
    The caller still owns current capture/identity, privilege and 1000ms checks.
    """

    def __init__(self, path, *, owner, run_id, game_pid, handoff_id=None, clock=None):
        if (not isinstance(owner, str) or owner not in OWNERS or not _text(run_id)
                or type(game_pid) is not int or game_pid <= 0):
            raise ValueError('unverified HIF ownership request')
        self.path, self.owner, self.run_id = Path(path).resolve(), owner, run_id
        self.game_pid, self.handoff_id, self.clock = game_pid, handoff_id, clock
        self._active, self._record, self._lock = False, None, None

    @property
    def record(self):
        if not self._active:
            raise RuntimeError('HIF ownership lease is not active')
        return deepcopy(self._record)

    @property
    def ownership_snapshot(self):
        """Detached current lease owner; no capture or input permission."""
        if not self._active:
            raise RuntimeError('HIF ownership lease is not active')
        return deepcopy(self._record['ownership'])

    @property
    def battle_entry_identity(self):
        """Read the original route identity and stamina limit for this published battle."""
        if self.owner != 'task275' or not self.authorise_input():
            return None
        try:
            record = self._record
            handoff = record['handoff']
            state = record['ledger']['state']
            prior = handoff['nonbattle_context']
            identity = prior['recovery_identity']
            number, revision = handoff['round_number'], handoff['ledger_revision']
            idol, plan, maximum = identity['idol'], identity['plan'], identity['max_stamina']
            if (record['ownership']['phase'] != 'battle' or handoff['to_owner'] != self.owner
                    or not (record['run_id'] == state['run_id'] == identity['run_id'] == self.run_id)
                    or prior['game_pid'] != self.game_pid or type(number) is not int or number not in (1, 2)
                    or not (self.handoff_id == handoff['handoff_id'] == f'{self.run_id}:r{number}:ledger-{revision}')
                    or type(revision) is not int or state['revision'] != revision or not _text(idol) or not _text(plan)
                    or type(maximum) is not int or maximum <= 0
                    or state['max_stamina'] != maximum or (idol, plan, maximum) != (
                        handoff['idol'], handoff['plan'], handoff['max_stamina'])):
                return None
            frames = prior['identity_frame_ids']
            if number == 1:
                start = prior['cultivation_start']
                request, creation = start['request'], start['creation_receipt']
                if (start['run_id'] != self.run_id or start['game_pid'] != self.game_pid
                        or start['status'] != 'complete' or request['identity'] != [idol, plan]
                        or request['source_revision'] != identity['revision'] or request['source_frame_ids'] != frames
                        or creation['resources']['max_stamina'] != maximum or handoff['prior_terminal_source'] is not None):
                    return None
                source, maximum_frames = 'cultivation_start', creation['frame_ids']
            else:
                scope, terminal = prior['recovery_identity_scope'], prior['terminal_receipt']
                if (scope['run_id'] != self.run_id or scope['game_pid'] != self.game_pid
                        or scope['prior_ledger_revision'] != prior['ledger_revision']
                        or prior['handoff_id'] != f'{self.run_id}:r1:ledger-{prior["ledger_revision"]}'
                        or not (scope['terminal_source'] == prior['terminal_source'] == handoff['prior_terminal_source'])
                        or not _text(scope['terminal_source']) or terminal['accepted'] is not True or terminal['terminal'] is not True
                        or (scope['round_number'], scope['live_run_id'], scope['live_revision']) != (
                            1, terminal['run_id'], terminal['revision']) or terminal['round_number'] != 1):
                    return None
                source, maximum_frames = 'recovery_identity', frames
            entry_frames = handoff['source_frame_ids']
            for keys in (entry_frames, frames, maximum_frames):
                if not has_single_frame_id(keys) or not re.fullmatch(rf'route-frame-{self.game_pid}-\d+', keys[0]):
                    return None
            entry_time = int(entry_frames[0].rsplit('-', 1)[1])
            if any(entry_time <= int(key.rsplit('-', 1)[1]) for key in (frames[0], maximum_frames[0])):
                return None
        except (KeyError, TypeError, ValueError):
            return None
        return dict(run_id=self.run_id, handoff_id=self.handoff_id, round_number=number, ledger_revision=revision,
            game_pid=self.game_pid, idol=idol, plan=plan, max_stamina=maximum, identity_source=source,
            source_frame_ids=list(entry_frames), identity_frame_ids=list(frames), max_stamina_frame_ids=list(maximum_frames))

    def __enter__(self):
        if self._active:
            raise RuntimeError('HIF ownership lease already active')
        if self.clock is None:
            from .host import windows_route_clock
            self.clock = windows_route_clock()
        self._lock = locked_hif_record(self.path)
        self._lock.__enter__()
        try:
            self._record = _read_owned_record(self.path, run_id=self.run_id, clock_epoch=self.clock.epoch)
            ownership = self._record['ownership']
            if ownership['owner'] != self.owner or ownership['game_pid'] != self.game_pid:
                raise ValueError('HIF input belongs to a different owner/game process')
            if self.owner == 'task275':
                handoff = self._record['handoff']
                if (ownership['phase'] != 'battle' or not _text(self.handoff_id)
                        or not isinstance(handoff, dict) or handoff.get('handoff_id') != self.handoff_id
                        or handoff.get('to_owner') != self.owner):
                    raise ValueError('HIF battle handoff is not acknowledged')
                self._require_settled()
            self._active = True
            return self
        except BaseException:
            self._lock.__exit__(None, None, None)
            self._lock = None
            raise

    def __exit__(self, *args):
        self._active = False
        self._record = None
        if self._lock is not None:
            self._lock.__exit__(*args)
            self._lock = None

    def authorise_input(self):
        return (self._active and self._record['ownership']['owner'] == self.owner
            and self._record['ownership']['game_pid'] == self.game_pid)

    def _require_settled(self):
        transition, ledger = self._record['transition'], self._record['ledger']
        if (transition['pending'] is not None or transition['stopped'] is not False
                or not isinstance(ledger, dict) or ledger.get('pending') is not None
                or not isinstance(ledger.get('state'), dict)):
            raise ValueError('HIF unresolved route/ledger cannot be handed over')

    def bind_nonbattle_ledger(self, ledger, *, frame_ids):
        """Reconcile recovery only from a complete, independently observed ledger."""
        from .ledger import HifLedger
        from .deck_history import record_deck_settlement
        if (not self.authorise_input() or self.owner != 'task600'
                or self._record['ownership']['phase'] != 'recovery'
                or self._record['transition']['pending'] is not None
                or self._record['transition']['stopped'] is not False
                or not isinstance(ledger, HifLedger) or ledger.pending is not None
                or ledger.state.run_id != self.run_id or not has_single_frame_id(frame_ids)):
            raise ValueError('HIF recovery needs a complete current nonbattle ledger')
        old = self._record['ledger']
        handoff = self._record['handoff'] or {}
        old_frames = (*handoff.get('source_frame_ids', ()),
            *handoff.get('terminal_receipt', {}).get('frame_ids', ()))
        if (set(frame_ids) & set(old_frames) or (old is not None and (
                old.get('pending') is not None or ledger.state.revision <= old['state']['revision']))):
            raise ValueError('HIF recovery cannot reuse old evidence or discard pending')
        candidate = deepcopy(self._record)
        candidate['ledger'] = ledger.record()
        candidate['ownership'].update(phase='nonbattle',
            revision=candidate['ownership']['revision'] + 1)
        candidate['handoff'] = dict(handoff, recovery_frame_ids=list(frame_ids))
        candidate['handoff'] = record_deck_settlement(candidate['handoff'], old['state'] if old is not None else None,
            candidate['ledger']['state'], source='deck_census', frames=frame_ids, census=True)
        write_hif_record(self.path, candidate)
        self._record = candidate

    def publish_handoff(self, handoff):
        from .hif import active_hif_role

        from .deck_history import record_exam_deck

        if (not self.authorise_input() or self.owner != 'task600'
                or self._record['ownership']['phase'] != 'nonbattle'):
            raise ValueError('only the active nonbattle owner can hand over')
        self._require_settled()
        previous = self._record['handoff'] or {}
        if previous.get('terminal_receipt', {}).get('round_number') == 2:
            raise ValueError('HIF round two already completed; no repeated battle handoff')
        required = {'handoff_id', 'to_owner', 'round_number', 'policy_profile', 'idol', 'plan',
            'max_stamina', 'source_frame_ids', 'prior_terminal_source', 'ledger_revision'}
        state = self._record['ledger']['state']
        if (not isinstance(handoff, dict) or set(handoff) != required
                or not _text(handoff['handoff_id']) or handoff['to_owner'] != 'task275'
                or type(handoff['round_number']) is not int or handoff['round_number'] not in (1, 2)
                or not isinstance(handoff['policy_profile'], str) or handoff['policy_profile'] not in FORMAL_PROFILES
                or (handoff['idol'], handoff['plan']) != active_hif_role(self._record)
                or handoff['max_stamina'] != state.get('max_stamina')
                or type(handoff['ledger_revision']) is not int
                or handoff['ledger_revision'] != state.get('revision')
                or state.get('run_id') != self.run_id or not has_single_frame_id(handoff['source_frame_ids'])
                or (handoff['round_number'] == 1 and handoff['prior_terminal_source'] is not None)
                or (handoff['round_number'] == 2 and not _text(handoff['prior_terminal_source']))):
            raise ValueError('incomplete or contradictory HIF battle handoff')
        if handoff['round_number'] == 1 and previous.get('terminal_receipt', {}).get('round_number') == 1:
            raise ValueError('HIF round one already completed; no repeated battle handoff')
        candidate = deepcopy(self._record)
        candidate['ownership'].update(owner='task275', phase='battle',
            revision=candidate['ownership']['revision'] + 1)
        candidate['handoff'] = deepcopy(handoff)
        if previous:
            # Keep the original failed navigation receipts and current
            # inventory evidence in this same envelope through the handoff.
            candidate['handoff']['nonbattle_context'] = deepcopy(previous)
        candidate['handoff'] = record_exam_deck(candidate['handoff'], state,
            round_number=handoff['round_number'], frame_ids=handoff['source_frame_ids'])
        write_hif_record(self.path, candidate)
        self._record = candidate

    def return_to_nonbattle(self, terminal_receipt, *, terminal_source=None, terminal_runtime=None):
        if not self.authorise_input() or self.owner != 'task275':
            raise ValueError('only the active battle owner can return')
        self._require_settled()
        if terminal_source is not None or terminal_runtime is not None:
            _verify_terminal_source(self._record, terminal_receipt, terminal_source, terminal_runtime)
        handoff = self._record['handoff']
        live = handoff.get('live_context')
        if (not isinstance(terminal_receipt, dict) or terminal_receipt.get('accepted') is not True
                or terminal_receipt.get('terminal') is not True
                or terminal_receipt.get('round_number') != handoff['round_number']
                or not isinstance(live, dict)
                or terminal_receipt.get('run_id') != live['run_id']
                or terminal_receipt.get('revision') != live['revision']
                or type(terminal_receipt.get('score')) is not int or terminal_receipt['score'] < 0
                or not valid_terminal_frame_ids(terminal_receipt.get('frame_ids'))
                or set(terminal_receipt['frame_ids']) & set(handoff['source_frame_ids'])):
            raise ValueError('HIF battle return lacks an independent terminal receipt')
        candidate = deepcopy(self._record)
        candidate['ownership'].update(owner='task600', phase='recovery',
            revision=candidate['ownership']['revision'] + 1)
        candidate['handoff']['terminal_receipt'] = deepcopy(terminal_receipt)
        # The battle host persists the full report before calling this API.
        # Publish its stable identity with the owner/receipt in one write, so
        # a crash before the outer route checkpoint cannot lose the source.
        if terminal_source is not None:
            candidate['handoff']['terminal_source'] = terminal_source
        # The pre-battle ledger is retained as historical data, not a current
        # baseline. Recovery must independently observe all post-battle fields.
        write_hif_record(self.path, candidate)
        self._record = candidate

    def bind_live_context(self, *, run_id, revision, idol, plan, max_stamina, round_number):
        if not self.authorise_input() or self.owner != 'task275':
            raise ValueError('only the active battle owner can bind Live')
        handoff = self._record['handoff']
        if (not _text(run_id) or type(revision) is not int or revision < 1
                or (idol, plan, max_stamina, round_number) != (
                    handoff['idol'], handoff['plan'], handoff['max_stamina'], handoff['round_number'])):
            raise ValueError('current HIF Live context contradicts its handoff')
        live = dict(run_id=run_id, revision=revision, idol=idol, plan=plan,
            max_stamina=max_stamina, round_number=round_number)
        if handoff.get('live_context') is not None and handoff['live_context'] != live:
            raise ValueError('HIF Live context is already bound; cannot replace it')
        candidate = deepcopy(self._record)
        candidate['handoff']['live_context'] = live
        write_hif_record(self.path, candidate)
        self._record = candidate
