"""Measured whole-world reserve for optional common-cohort validation rounds."""
from __future__ import annotations

import math
import time

VERSION = 'hif-uniform-validation-time-guard-v2'
MINIMUM_ROUNDS = 4
DELIVERY_RESERVE_SECONDS = 1.
WINDOW_LIMIT_SECONDS = 30.


class UniformValidationTimeGuard:
    """Only complete rounds justify a reserve; the hard deadline still wins."""

    def __init__(self, *, capacity, root_count, retained_present, deadline,
                 window_limit_seconds=WINDOW_LIMIT_SECONDS):
        if (type(window_limit_seconds) not in (int, float)
                or not math.isfinite(window_limit_seconds) or window_limit_seconds <= 0):
            raise ValueError('mcts_uniform_validation_invalid_window_limit')
        self.deadline = deadline
        self.started = time.perf_counter()
        self.report = dict(version=VERSION, minimum_rounds=MINIMUM_ROUNDS,
            delivery_reserve_seconds=DELIVERY_RESERVE_SECONDS,
            window_limit_seconds=window_limit_seconds, capacity=capacity,
            root_count=root_count, retained_present=retained_present,
            rounds=[], prestart_checks=[], completed_rounds=0,
            max_completed_round_seconds=0., minimum_coverage=False,
            complete=False, stop_reason='not_run', elapsed_seconds=0.,
            deadline_remaining_seconds=deadline - self.started,
            finished_before_deadline=False, timing_scope='SERIAL_WHOLE_ROUND')

    def can_start(self, index):
        now = time.perf_counter()
        if now >= self.deadline:
            raise TimeoutError('mcts_uniform_validation_deadline')
        report = self.report
        if report['completed_rounds'] < MINIMUM_ROUNDS:
            return True
        required = report['max_completed_round_seconds'] + DELIVERY_RESERVE_SECONDS
        remaining = self.deadline - now
        allowed = remaining > required
        report['prestart_checks'].append(dict(next_round_index=index,
            completed_rounds=report['completed_rounds'], remaining_seconds=remaining,
            max_completed_round_seconds=report['max_completed_round_seconds'],
            required_seconds=required, allowed=allowed))
        if not allowed:
            report['stop_reason'] = 'time_reserve_reached'
        return allowed

    def record_round(self, *, index, seed, started, root_started, root_completed,
                     retained_started, retained_completed, complete):
        seconds = (time.perf_counter() - started
            if self.report['timing_scope'] == 'SERIAL_WHOLE_ROUND' else None)
        self.report['rounds'].append(dict(index=index, planning_seed=seed,
            elapsed_seconds=seconds, root_started=root_started, root_completed=root_completed,
            retained_started=retained_started, retained_completed=retained_completed,
            complete=complete))
        if complete:
            self.report['completed_rounds'] += 1
            if seconds is not None:
                self.report['max_completed_round_seconds'] = max(
                    self.report['max_completed_round_seconds'], seconds)

    def finish(self, *, complete, reason):
        report = self.report
        report['minimum_coverage'] = report['completed_rounds'] >= MINIMUM_ROUNDS
        report['complete'] = bool(complete and report['minimum_coverage'])
        if not report['complete'] or report['stop_reason'] != 'time_reserve_reached':
            report['stop_reason'] = reason or 'completed'
        now = time.perf_counter()
        report['elapsed_seconds'] = now - self.started
        report['deadline_remaining_seconds'] = self.deadline - now
        report['finished_before_deadline'] = now < self.deadline
        if not report['finished_before_deadline']:
            report.update(complete=False, stop_reason='timeout')
        return report['complete']
