"""Use the existing game input lock and native battle receipts on a live battle page."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from route.ownership import locked_hif_record, valid_terminal_frame_ids

from .hif_recovery_reports import read_recovery_business_report


def _battle_id(run_id, number):
    return f'{run_id}:current-battle:r{number}'


def _reports(runtime, run_id, game_pid):
    # Current-page attempts are named by the entry point. Old route and
    # unrelated cultivation reports need not be opened, even on cold restart.
    for path in sorted(Path(runtime).glob(f'result-*-{run_id}-current-battle-live*.json')):
        report = read_recovery_business_report(path)
        if report.get('endpoints', {}).get('game', {}).get('pid') == game_pid:
            yield report


def current_battle_round(runtime, run_id, game_pid):
    """Return the last locally started, still unfinished round for this game."""
    started, finished = set(), set()
    for report in _reports(runtime, run_id, game_pid):
        for number in (1, 2):
            if report.get('handoff_id') == _battle_id(run_id, number):
                started.add(number)
                if report.get('status') == 'TERMINAL_VERIFIED':
                    finished.add(number)
    return max(started - finished, default=None)


class CurrentBattleSession:
    """The current process holds the route byte lock; the route record stays read-only."""

    current_screen = True
    owner = 'task275'
    battle_entry_identity = None

    def __init__(self, path, *, runtime, run_id, game_pid, round_number, policy_profile, plan, authorise_task=None):
        self.path, self.runtime = Path(path), Path(runtime)
        self.run_id, self.game_pid = run_id, game_pid
        self.handoff_id = _battle_id(run_id, round_number)
        self._authorise_task = authorise_task
        self._lock = None
        self._record = dict(run_id=run_id, ledger=None, transition=dict(pending=None, stopped=False), ownership=dict(
            owner=self.owner, phase='battle', revision=1, game_pid=game_pid),
            handoff=dict(source='current_battle_page', to_owner=self.owner, handoff_id=self.handoff_id,
                round_number=round_number, policy_profile=policy_profile, plan=plan,
                ledger_revision=None))

    def __enter__(self):
        if self._lock is not None:
            raise RuntimeError('current battle session already has the input lock')
        lock = locked_hif_record(self.path)
        lock.__enter__()
        self._lock = lock
        try:
            if self.path.exists():
                route = json.loads(self.path.read_bytes())
                owner = route['ownership']
                if route['run_id'] != self.run_id or owner['game_pid'] != self.game_pid:
                    raise ValueError('current battle no longer belongs to the same game')
                if owner['owner'] == 'task275' and owner['phase'] == 'battle':
                    raise ValueError('current battle already has a published route session')
            for report in _reports(self.runtime, self.run_id, self.game_pid):
                if report.get('handoff_id') == self.handoff_id:
                    self._restore(report)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        lock, self._lock = self._lock, None
        if lock is not None:
            lock.__exit__(*args)

    @property
    def record(self):
        if self._lock is None:
            raise RuntimeError('current battle input lock is not active')
        return deepcopy(self._record)

    @property
    def ownership_snapshot(self):
        return self.record['ownership']

    @property
    def completed(self):
        return self._record['handoff'].get('terminal_receipt') is not None

    def authorise_input(self):
        return (self._lock is not None and not self.completed
            and (self._authorise_task is None or self._authorise_task() is True))

    def bind_live_context(self, *, run_id, revision, idol, plan, max_stamina, round_number):
        handoff = self._record['handoff']
        live = dict(run_id=run_id, revision=revision, idol=idol, plan=plan,
            max_stamina=max_stamina, round_number=round_number)
        if ((plan, round_number) != (handoff['plan'], handoff['round_number'])
                or handoff.get('live_context', live) != live):
            raise ValueError('current battle observation changed its live identity')
        handoff['live_context'] = live

    def _restore(self, report):
        handoff = self._record['handoff']
        opening = report.get('handoff_opening')
        if opening is None:
            return
        live = report.get('observed_live_context') or {}
        if ((opening.get('route_run_id'), opening.get('handoff_id'), opening.get('round_number'),
                opening.get('policy_profile'), opening.get('opening_inventory_source'),
                opening.get('live_run_id'), opening.get('live_revision')) != (
                self.run_id, self.handoff_id, handoff['round_number'], handoff['policy_profile'],
                'current_complete_opening', live.get('run_id'), live.get('revision'))):
            raise ValueError('current battle report belongs to another opening')
        self.bind_live_context(**live, round_number=opening['round_number'])
        if report.get('status') == 'TERMINAL_VERIFIED':
            receipt = report.get('terminal_settlement') or {}
            if (report.get('transaction_closed') is not True or report.get('live_return') is not True
                    or receipt.get('accepted') is not True or receipt.get('terminal') is not True
                    or type(receipt.get('score')) is not int or receipt['score'] < 0
                    or not valid_terminal_frame_ids(receipt.get('frame_ids'))
                    or (receipt.get('run_id'), receipt.get('revision'), receipt.get('round_number'))
                        != (live['run_id'], live['revision'], opening['round_number'])):
                raise ValueError('current battle terminal report is incomplete')
            handoff.update(terminal_receipt=deepcopy(receipt), terminal_source=report['id'])

    def return_to_nonbattle(self, receipt, *, terminal_source, terminal_runtime):
        if not self.authorise_input() or Path(terminal_runtime).resolve() != self.runtime.resolve():
            raise ValueError('current battle terminal source is outside its active session')
        path = self.runtime / f'result-{terminal_source}.json'
        report = read_recovery_business_report(path)
        if (report.get('handoff_id') != self.handoff_id or report.get('status') != 'TERMINAL_VERIFIED'
                or report.get('endpoints', {}).get('game', {}).get('pid') != self.game_pid
                or report.get('terminal_settlement') != receipt or not report.get('handoff_opening')):
            raise ValueError('current battle terminal differs from its saved result')
        self._restore(report)
