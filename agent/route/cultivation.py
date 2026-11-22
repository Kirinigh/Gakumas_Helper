"""One Maa invocation across independently confirmed route/battle boundaries.

Only a trusted client can bind this runtime. Saved budgets and task parameters
cannot grant input, create a cultivation, or turn a STOP into an automatic retry.
"""
from __future__ import annotations

import os
import re
import json
from copy import deepcopy
from pathlib import Path

from .hif import active_hif_role
from .engine import Outcome, Decision, has_single_frame_id, source_frames_match, validate_observation
from .ownership import HifInputSession, write_hif_record, locked_hif_record, validate_ownership

CONTINUE_BOUNDARIES = frozenset({'hif_current_initial_complete_ledger_observed',
    'hif_current_initial_partial_ledger_observed',
    'hif_new_cultivation_created',
    'hif_new_cultivation_created_actual_ap_receipt_required',
    'hif_current_complete_nonbattle_ledger_observed', 'hif_actual_start_ap_verified_resume_current_save',
    'hif_observed_training_reward_complete', 'hif_observed_training_reward_census_complete', 'hif_observed_activity_reward_census_complete',
    'hif_observed_lesson_support_grant_complete',
    'hif_observed_customization_census_complete',
    'hif_verified_day_seven_route_boundary',
    'sequence_action_budget_reached'})
MAX_WINDOWS = 64
MAX_ELAPSED_MS = 21_600_000


def _budget_container(handoff, *, create=False):
    current = handoff
    for _ in range(20):
        if not isinstance(current, dict):
            break
        if 'automatic_cultivation' in current:
            return current
        current = current.get('nonbattle_context')
    else:
        raise ValueError('automatic cultivation history nesting is invalid')
    if not create:
        return None
    return handoff.setdefault('nonbattle_context', {})


def _progress(record):
    from .new_run import start_record
    state = (record.get('ledger') or {}).get('state') or {}
    start = start_record(record.get('handoff')) or {}
    return record['ownership']['revision'], record['ownership']['phase'], state.get('revision'), state.get('phase'), start.get('status')


class HifCultivationRuntime:
    def __init__(self, *, context, path, run_id, game_pid, runtime,
            authorise_capture, authorise_input, authorise_battle, endpoint, evidence_sink=None,
            bind_nonbattle=None, battle_service=None, clock=None, approved_budget_migration=None,
            authorise_preparation=None, authorise_navigation_preparation=None):
        from .host import windows_route_clock
        if (not isinstance(run_id, str) or re.fullmatch(r'[a-z0-9-]+', run_id) is None
                or type(game_pid) is not int or game_pid < 1
                or any(not callable(value) for value in (authorise_capture, authorise_input, authorise_battle, endpoint))):
            raise ValueError('automatic cultivation needs a trusted run, endpoint and separate input grants')
        if authorise_preparation is not None and not callable(authorise_preparation):
            raise ValueError('automatic cultivation preparation permission is not callable')
        if authorise_navigation_preparation is not None and not callable(authorise_navigation_preparation):
            raise ValueError('automatic cultivation navigation preparation permission is not callable')
        self.context, self.path, self.run_id, self.game_pid = context, Path(path), run_id, game_pid
        self.runtime, self.clock = Path(runtime), clock or windows_route_clock()
        self.pause_at_battle = False
        self.authorise_capture, self.authorise_input, self.authorise_battle = authorise_capture, authorise_input, authorise_battle
        self.authorise_preparation = authorise_preparation
        self.authorise_navigation_preparation = authorise_navigation_preparation
        self.endpoint, self.evidence_sink = endpoint, evidence_sink
        self.bind_nonbattle, self.battle_service = bind_nonbattle, battle_service
        self.approved_budget_migration = approved_budget_migration
        self.bound = None
        self.terminal_report = None
        self.report = dict(status='STOP', run_id=run_id, windows=[], battle_outcomes=[],
            input_count=0, input_success_count=0, input_uncertain_count=0, reason=None)

    def _read(self):
        record = json.loads(self.path.read_text(encoding='utf-8'))
        owner = validate_ownership(record['ownership'])
        if record.get('schema_version') != 3 or record.get('run_id') != self.run_id or owner['game_pid'] != self.game_pid:
            raise ValueError('automatic cultivation run or target process changed')
        return record

    def _checkpoint(self, *, opening=False, reason=None, complete=False):
        with locked_hif_record(self.path):
            record = self._read()
            from .terminal_source import bind_terminal_return
            candidate, prepared_report = bind_terminal_return(record, self.runtime)
            candidate = deepcopy(candidate)
            if prepared_report is not None:
                self.terminal_report = prepared_report
            handoff = candidate['handoff']
            if handoff is None:
                handoff = candidate['handoff'] = {}
            container = _budget_container(handoff, create=True)
            budget = container.get('automatic_cultivation')
            now = self.clock.now()
            if budget is None:
                budget = dict(schema_version=3, run_id=self.run_id, game_pid=self.game_pid,
                    clock_epoch=self.clock.epoch, started_at_ms=now, last_at_ms=now,
                    windows_used=0,
                    status='running', initial_process_id=os.getpid(), process_ids=[os.getpid()], terminal_source=None,
                    active_elapsed_ms=0, paused_at_ms=None, pause_receipts=[], last_process_id=os.getpid(), accounting_state='active',
                    initial_day=((record.get('ledger') or {}).get('state') or {}).get('day'),
                    initial_phase=((record.get('ledger') or {}).get('state') or {}).get('phase'))
            else:
                budget = deepcopy(budget)
            if budget.get('schema_version') == 1 and self.approved_budget_migration is not None:
                plan = self.approved_budget_migration
                proposed = plan.get('budget') or {}
                if (plan.get('kind') != 'legacy_active_runtime_migration_preview'
                        or not isinstance(plan.get('expected_host_record'), dict) or plan['expected_host_record'] != record
                        or proposed.get('legacy_migration', {}).get('original') != budget
                        or any(proposed.get(key) != budget.get(key) for key in (
                            'run_id', 'game_pid', 'clock_epoch', 'started_at_ms', 'windows_used',
                            'max_windows', 'max_elapsed_ms', 'initial_process_id', 'process_ids',
                            'initial_day', 'initial_phase', 'terminal_source'))):
                    raise ValueError('approved budget migration no longer matches the untouched original record')
                budget = deepcopy(proposed)
            from .run_budget import advance_budget, validate_budget
            validate_budget(budget, run_id=self.run_id, game_pid=self.game_pid, epoch=self.clock.epoch,
                now=now)
            if budget['status'] == 'complete':
                return record, budget
            from .nonbattle_date import navigation_closed
            budget = advance_budget(budget, now=now, process_id=os.getpid(), opening=opening, reason=reason, complete=complete,
                pause_permitted=record['transition']['pending'] is None and (record.get('ledger') or {}).get('pending') is None
                    and navigation_closed(record))
            if self.terminal_report is not None:
                budget['terminal_source'] = self.terminal_report['id']
            container['automatic_cultivation'] = budget
            write_hif_record(self.path, candidate)
            return candidate, budget

    def _size(self):
        endpoint = self.endpoint()
        game = endpoint['game']
        if (game['pid'] != self.game_pid or not endpoint
                or any(item.get('integrity_rid') != 0x3000 for item in endpoint.values())):
            raise ValueError('automatic cultivation target endpoint changed')
        size = game['observed_client_size']
        if (not isinstance(size, (tuple, list)) or len(size) != 2
                or any(type(value) is not int or value < 1 for value in size)):
            raise ValueError('automatic cultivation current client dimensions are unresolved')
        return tuple(size)

    def _bind(self, size):
        factory = self.bind_nonbattle
        if factory is None:
            from .production import bind_hif_nonbattle_runtime
            factory = bind_hif_nonbattle_runtime
        preparation = (dict(authorise_preparation=self.authorise_preparation)
            if self.authorise_preparation is not None else {})
        self.bound = factory(self.context, path=self.path, run_id=self.run_id, game_pid=self.game_pid,
            client_size=size, authorise_capture=self.authorise_capture, authorise_input=self.authorise_input,
            evidence_sink=self.evidence_sink, terminal_report=self.terminal_report, terminal_runtime=self.runtime,
            **preparation)

    def _handoff(self, size, ordinal, *, battle_policy=None):
        # host.run has already returned and released its lease. Acquire a new
        # lease for fresh source proof and atomic ownership publication only.
        host = self.bound.gkh_hif_route_host
        if host._active or self.authorise_battle() is not True or self.authorise_capture() is not True:
            raise ValueError('automatic battle handoff lacks its separate current grant or released route lease')
        with HifInputSession(self.path, owner='task600', run_id=self.run_id, game_pid=self.game_pid, clock=self.clock) as session:
            host._load()
            page = self.bound.gkh_hif_route_producer.observe()
            record = session.record
            from .battle_policy import retained_profile
            profile = retained_profile(record.get('handoff'), selection=battle_policy)
            previous = (record['handoff'] or {}).get('terminal_receipt')
            round_number = 1 if previous is None else 2
            if previous is not None and (previous.get('round_number') != 1 or self.terminal_report is None
                    or self.terminal_report.get('terminal_settlement') != previous):
                raise ValueError('automatic round two lacks its original round-one terminal source')
            seen = page.observation
            idol, plan = active_hif_role(host)
            if (seen.scene != 'battle_entry' or seen.run_id != self.run_id or seen.day != 7
                    or (seen.idol, seen.plan, seen.scenario) != (idol, plan, 'HIF')
                    or validate_observation(seen) is not None or not source_frames_match(seen, page.frame_ids)
                    or page.baseline != host.ledger.state or host.ledger.pending is not None
                    or not 0 <= seen.capture_age_ms + self.clock.now() - seen.revision <= 1000):
                raise ValueError('automatic battle handoff lacks a fresh complete current entry')
            handoff_id = f'{self.run_id}:r{round_number}:ledger-{host.ledger.state.revision}'
            session.publish_handoff(dict(handoff_id=handoff_id, to_owner='task275', round_number=round_number,
                policy_profile=profile, idol=idol, plan=plan,
                max_stamina=host.ledger.state.max_stamina, source_frame_ids=list(page.frame_ids),
                prior_terminal_source=None if round_number == 1 else self.terminal_report['id'],
                ledger_revision=host.ledger.state.revision))
            published = session.record
        from live_decision.hif_battle_entry import HifBattleRequest
        return HifBattleRequest.from_record(published,
            attempt_id=f'{ordinal:04d}-{self.run_id}-r{round_number}', client_size=size)

    def _battle(self, request):
        if self.authorise_battle() is not True:
            raise ValueError('automatic battle input is not authorised for this run')
        service = self.battle_service
        if service is None:
            from live_decision.hif_battle_entry import run_hif_battle
            service = run_hif_battle
        # Original Maa context, with no old route lease or battle input guard.
        outcome = service(self.context, request=request, path=self.path, runtime=self.runtime, authorised_run_id=self.run_id)
        self.report['battle_outcomes'].append(outcome)
        self.report.update(input_success_count=None, input_uncertain_count=None)
        if type(outcome.get('input_count')) is not int or outcome['input_count'] < 0:
            self.report['input_count'] = None
            return Decision(Outcome.STOP, 'automatic_battle_input_accounting_unknown')
        self.report['input_count'] += outcome['input_count']
        if outcome.get('status') != 'TERMINAL_VERIFIED' or outcome.get('returned_to_nonbattle') is not True:
            if outcome.get('status') == 'STOP' and outcome.get('returned_to_nonbattle') is False:
                from .battle_date import recover_battle_date
                if recover_battle_date(self, request, outcome):
                    return None
            return Decision(Outcome.STOP, outcome.get('stop_reason') or 'automatic_battle_not_verified_returned')
        after = self._read()
        terminal = outcome.get('terminal_report')
        if (after['ownership'] != dict(owner='task600', phase='recovery', game_pid=self.game_pid,
                revision=request.ownership_revision + 1)
                or not isinstance(terminal, dict) or terminal.get('status') != 'TERMINAL_VERIFIED'
                or terminal.get('returned_to_nonbattle') is not True
                or after['handoff'].get('handoff_id') != request.handoff_id
                or after['handoff'].get('terminal_receipt') != terminal.get('terminal_settlement')
                or after['handoff'].get('round_number') != request.round_number):
            raise ValueError('automatic battle return differs from its shared terminal receipt')
        self.terminal_report, self.bound = terminal, None
        return None

    def _reserve_pending_readback(self, reason):
        # One separate, read-only late receipt attempt for a known successful
        # dispatch. This is not a general STOP retry or a renewed old deadline.
        if reason not in ('transition_timeout', 'ordinary HIF observation deadline reached',
                'public-course reward observation reached its original deadline', 'activity original observation deadline reached',
                'HIF current recovery capture deadline reached'):
            return False
        from types import SimpleNamespace

        from .checkpoint import restore_transition
        from .input_receipt import original_input_report
        from .activity_production import DESTINATIONS
        with locked_hif_record(self.path):
            record = self._read()
            gate = restore_transition(record['transition'], clock_epoch=self.clock.epoch)
            if (record['ownership']['owner'] != 'task600' or gate.pending is None
                    or gate.pending.target.kind not in (*DESTINATIONS, 'round_one_result_next', 'round_one_award_next', 'recovery_drink_detail',
                        'recovery_drink_close', 'confirm_training', 'training_reward_ack', 'training_reward_upgrade_ack',
                        'recovery_inventory_detail', 'recovery_inventory_scroll')):
                return False
            carrier = (record.get('handoff') or {}).get('nonbattle_context', {}).get('route_input_receipt')
            if not isinstance(carrier, dict) or carrier.get('recovery_reads') != 0:
                return False
            try:
                original_input_report(SimpleNamespace(handoff=record['handoff'], run_id=self.run_id,
                    game_pid=self.game_pid, clock=self.clock), gate)
            except ValueError as error:
                self.report.setdefault('readonly_recovery_rejections', []).append(str(error))
                return False
            candidate = deepcopy(record)
            saved = candidate['handoff']['nonbattle_context']['route_input_receipt']
            saved.update(recovery_reads=1, original_stop_reason=reason)
            write_hif_record(self.path, candidate)
        self.report.setdefault('readonly_recoveries', []).append(dict(target_kind=gate.pending.target.kind,
            issued_at_ms=gate.issued_at_ms, original_timeout_ms=gate.timeout_ms, reason=reason, business_input_replayed=False))
        return True

    def _confirmed_reward_reservation(self, after, original, report):
        """Recognize one existing cost held across already confirmed free steps."""
        from types import SimpleNamespace

        from . import rewards, activity_reward
        from .ledger import restore_ledger
        from .checkpoint import restore_transition
        host = self.bound.gkh_hif_route_host
        ledger = restore_ledger(after['ledger'], card_catalog=host._catalog, drink_ids=host._drink_ids)
        gate = restore_transition(after['transition'], clock_epoch=self.clock.epoch)
        carrier = SimpleNamespace(handoff=after['handoff'], ledger=ledger, ownership=after['ownership'],
            run_id=self.run_id, game_pid=self.game_pid)
        activity, course = activity_reward.request(carrier), rewards.request(carrier)
        if (activity is None) == (course is None):
            raise ValueError('reserved continuation needs exactly one original reward family')
        value = activity if activity is not None else course
        if value.get('reservation') != after['ledger']['pending']:
            raise ValueError('reserved continuation changed its original exact reservation')
        pending, proof = original.pending, None
        if activity is not None:
            if (not activity_reward.permits_navigation(carrier, ledger, gate)
                    or pending.target.kind not in (*activity_reward.KINDS, 'execute_outing', 'confirm_consult')):
                raise ValueError('activity is not an already confirmed free-navigation stage')
            proof = value.get('last_destination')
            if (not isinstance(proof, dict) or proof.get('kind') not in pending.expected_scenes
                    or (value['stage'] != 'census' and proof['kind'] != value['stage'])
                    or value['stage'] == 'census' and proof['kind'] not in (
                        'activity_inventory', 'activity_home', 'consult_shop', 'activity_drink_detail')
                    or proof.get('day') not in (value['day'], value['census_day'])):
                raise ValueError('activity continuation lacks its confirmed original destination')
            family = 'activity_reward'
        else:
            if not rewards.permits_navigation(carrier, ledger, gate):
                raise ValueError('course is not an already confirmed reward-navigation stage')
            kind = pending.target.kind
            if kind == 'confirm_training' and value['stage'] == 'awaiting_ack':
                proof, scene = value.get('result'), 'training_result'
            elif kind in ('training_reward_ack', 'training_reward_upgrade_ack') and value['stage'] == 'upgrade':
                proof, scene = value.get('upgrade'), 'reward_upgrade'
            elif kind in ('training_reward_ack', 'training_reward_upgrade_ack', 'reward_delete_result_ack', 'recovery_close') and value['stage'] == 'census':
                proof, scene = value.get('home'), 'reward_home'
            elif kind in ('recovery_inventory', 'recovery_inventory_detail', 'recovery_inventory_scroll') and value['stage'] == 'census':
                inventory = value.get('inventory') or {}
                proof, scene = dict(frame_ids=inventory.get('review_frame_ids')), 'reward_inventory'
            elif value['stage'] == 'deletion':
                deletion = value.get('deletion') or {}
                if kind in ('training_reward_ack', 'training_reward_upgrade_ack', 'finish_consult') and deletion.get('status') == 'active':
                    proof, scene = deletion, 'reward_delete_choice'
                elif kind in ('reward_delete_open', 'reward_delete_confirm') and deletion.get('status') == 'committed_pending_census':
                    proof, scene = deletion.get('result'), 'reward_delete_result'
                elif kind in ('reward_delete_select', 'reward_delete_scroll', 'reward_delete_open') and deletion.get('status') == 'active':
                    menu = 'delete_selected' if kind == 'reward_delete_select' else 'delete_confirm'
                    if kind == 'reward_delete_scroll' and (deletion.get('last_menu') or {}).get('kind') in ('delete_choice', 'delete_search'):
                        menu = deletion['last_menu']['kind']
                    if (kind == 'reward_delete_select' and pending.source.scene == 'reward_delete_choice'
                            and (deletion.get('last_menu') or {}).get('kind') == 'delete_reselect'):
                        menu = 'delete_reselect'
                    scene = 'reward_' + menu
                    proof = deletion.get('last_menu')
                    if (proof or {}).get('kind') != menu:
                        raise ValueError('course deletion continuation lacks its confirmed menu')
                else:
                    raise ValueError('course continuation does not match its saved deletion')
            else:
                raise ValueError('course continuation is not its original confirmed free step')
            if not isinstance(proof, dict) or scene not in pending.expected_scenes:
                raise ValueError('course continuation lacks its confirmed original destination')
            family = 'training_reward'
        source_frames = [row.get('frame_id') for row in report.get('frames', ())]
        check_frames = activity_reward.frames if activity is not None else rewards.reward_frames
        frames = check_frames(proof, (*source_frames, *value['source_frame_ids']))
        return dict(family=family, stage=value['stage'], transaction_id=value['transaction_id'], receipt_frame_ids=list(frames))

    def _run_date_navigation(self, ordinal):
        from .nonbattle_date import NonbattleDateNavigation
        # Discard observations collected before the date interruption, while
        # retaining the original durable business gate and receipt verbatim.
        self.bound = None
        self._bind(self._size())
        from .production import HifRecoveryProducer
        producer = getattr(self.bound, 'gkh_hif_route_producer', None)
        if not isinstance(producer, HifRecoveryProducer):
            raise ValueError('date navigation requires a fresh original production binding')
        child = NonbattleDateNavigation(self, producer)
        decision = None
        try:
            decision = child.run(timeout_ms=None)
        finally:
            attempted = [row for row in child.input_events if row.get('dispatch_attempted') is True]
            self.report['input_count'] += len(attempted)
            if type(self.report['input_success_count']) is int:
                self.report['input_success_count'] += sum(row.get('success') is True for row in attempted)
            if type(self.report['input_uncertain_count']) is int:
                self.report['input_uncertain_count'] += sum(row.get('uncertain') is True for row in attempted)
            self.report.setdefault('date_navigation', []).append(dict(ordinal=ordinal,
                decision=decision.reason if decision is not None else 'date_navigation_exception',
                input_events=deepcopy(child.input_events)))
        if decision.reason != 'nonbattle_date_navigation_complete_recheck_original_business':
            raise ValueError(decision.reason)
        self.bound = None

    def _record_overlay_inputs(self, steps):
        issued = [step for step in steps if step.get('result') in ('sent', 'uncertain')]
        if type(self.report['input_count']) is int:
            self.report['input_count'] += len(issued)
        if type(self.report['input_success_count']) is int:
            self.report['input_success_count'] += sum(step['result'] == 'sent' for step in issued)
        if type(self.report['input_uncertain_count']) is int:
            self.report['input_uncertain_count'] += sum(step['result'] == 'uncertain' for step in issued)

    def run(self, config):
        from live_decision.hif_battle_entry import HifBattleRequest

        from .new_run import start_record
        if config.mode != 'advance':
            raise ValueError('automatic cultivation requires the separately bound advance runtime')
        from .battle_policy import retained_profile
        try:
            retained_profile(self._read().get('handoff'), selection=config.battle_policy)
        except ValueError as error:
            self.report.update(status='STOP', reason=str(error))
            return Decision(Outcome.STOP, str(error))
        reason = None
        try:
            while True:
                retained_profile(self._read().get('handoff'), selection=config.battle_policy)
                if self.context.tasker.stopping or self.authorise_capture() is not True or self.authorise_input() is not True:
                    raise ValueError('automatic cultivation stopped or current permission revoked')
                record, budget = self._checkpoint(opening=True)
                self.report['initial_entry'] = dict(day=budget.get('initial_day'), phase=budget.get('initial_phase'),
                    process_ids=list(budget['process_ids']))
                if budget['status'] == 'complete':
                    self.report.update(status='COMPLETE', reason='hif_cultivation_complete_no_repeat')
                    return Decision(Outcome.PAUSE, 'hif_cultivation_complete_no_repeat')
                if self.terminal_report is None and budget.get('terminal_source'):
                    source = budget['terminal_source']
                    if re.fullmatch(r'[0-9]+-[a-z0-9-]+', source) is None:
                        raise ValueError('automatic terminal source identity is invalid')
                    self.terminal_report = json.loads((self.runtime / f'result-{source}.json').read_text(encoding='utf-8'))
                size = self._size()
                if record['ownership']['owner'] == 'task275':
                    if self.pause_at_battle:
                        reason = 'hif_battle_handoff_ready_for_task275'
                        break
                    from .battle_date import pending_navigation, recover_battle_date
                    if pending_navigation(record):
                        request = HifBattleRequest.from_identity(record,
                            attempt_id=f'{budget["windows_used"]:04d}-{self.run_id}-battle-date', client_size=size)
                        if not recover_battle_date(self, request):
                            raise ValueError('persisted battle navigation was not independently completed')
                        continue
                    date = record['handoff'].get('nonbattle_context', {}).get('date_recovery')
                    if record['handoff'].get('live_context') is None and date is None:
                        from live_decision.hif_battle_entry import load_battle_navigation_source

                        request = HifBattleRequest.from_identity(record,
                            attempt_id=f'{budget["windows_used"]:04d}-{self.run_id}-battle-date', client_size=size)
                        try:
                            source = load_battle_navigation_source(self.runtime, request, record)
                        except ValueError:
                            # A first handoff has no previous opening STOP. Its
                            # original Native admission still verifies history.
                            source = None
                        if source is not None:
                            if recover_battle_date(self, request):
                                continue
                            reason = source['source_report'].get('error') or 'original_battle_opening_stop'
                            break
                    request = HifBattleRequest.from_record(record,
                        attempt_id=f'{budget["windows_used"]:04d}-{self.run_id}-battle-resume', client_size=size)
                    decision = self._battle(request)
                    if decision is not None:
                        reason = decision.reason
                        break
                    continue
                if self.bound is None:
                    self._bind(size)
                host = self.bound.gkh_hif_route_host
                from .production import HifRecoveryProducer
                producer = getattr(self.bound, 'gkh_hif_route_producer', None)
                if isinstance(producer, HifRecoveryProducer):
                    producer.terminal_report, producer.terminal_runtime = self.terminal_report, self.runtime
                    host._ending_terminal_report = self.terminal_report
                from .nonbattle_date import KEY as DATE_KEY
                from .nonbattle_date import NonbattleDateObserved
                from .produce_settings import ProduceSettingsObserved
                from .candidate_pool import CandidatePoolNavigation, CandidatePoolObserved, pending_candidate_pool
                from .settings_navigation import SettingsNavigation, pending_settings
                self.bound.gkh_hif_cultivation_runtime = self
                if (pending_settings(record) or {}).get('input') is not None:
                    settings = SettingsNavigation(self).run()
                    self.report.setdefault('produce_settings', []).extend(settings)
                    self._record_overlay_inputs(settings)
                    continue
                if pending_candidate_pool(record) is not None:
                    pool = CandidatePoolNavigation(self).run()
                    self.report.setdefault('candidate_pool', []).append(pool)
                    self._record_overlay_inputs(pool['inputs'])
                    continue
                date_state = (record.get('handoff') or {}).get('nonbattle_context', {}).get(DATE_KEY)
                if isinstance(date_state, dict) and date_state.get('status') == 'active':
                    if not isinstance(producer, HifRecoveryProducer):
                        raise ValueError('date navigation lacks the original bound production observer')
                    self._run_date_navigation(budget['windows_used'])
                    continue
                offset = len(host.input_events)
                if isinstance(producer, HifRecoveryProducer):
                    producer.observation_window(record, self.clock.now())
                decision, route_error = None, None
                try:
                    decision = host.run(self.bound, config, max_actions=None, timeout_ms=None)
                except ValueError as error:
                    route_error = error
                finally:
                    inputs = host.input_events[offset:]
                    attempted = [item for item in inputs if item.get('dispatch_attempted') is True]
                    self.report['input_count'] += len(attempted)
                    if type(self.report['input_success_count']) is int:
                        self.report['input_success_count'] += sum(item.get('success') is True for item in attempted)
                    if type(self.report['input_uncertain_count']) is int:
                        self.report['input_uncertain_count'] += sum(item.get('uncertain') is True for item in attempted)
                    self.report['windows'].append(dict(ordinal=budget['windows_used'],
                        decision=decision.reason if decision is not None else 'nonbattle_exception',
                        input_count=len(attempted), input_events=deepcopy(inputs), ownership=deepcopy(host.ownership)))
                after = self._read()
                start = start_record(after.get('handoff')) or {}
                if 'initial_deck' in start and 'initial_deck' not in self.report:
                    self.report['initial_deck'] = deepcopy(start['initial_deck'])
                if any(item.get('uncertain') is True or item.get('success') is not True for item in attempted):
                    reason = 'automatic_nonbattle_input_not_confirmed'
                    break
                if route_error is not None:
                    if isinstance(route_error, CandidatePoolObserved):
                        pool = CandidatePoolNavigation(self).run(route_error.request)
                        self.report.setdefault('candidate_pool', []).append(pool)
                        self._record_overlay_inputs(pool['inputs'])
                        continue
                    if isinstance(route_error, ProduceSettingsObserved):
                        settings = SettingsNavigation(self).run()
                        self.report.setdefault('produce_settings', []).extend(settings)
                        self._record_overlay_inputs(settings)
                        continue
                    if isinstance(route_error, NonbattleDateObserved):
                        if not isinstance(producer, HifRecoveryProducer):
                            raise ValueError('date navigation lacks the original bound production observer')
                        self._run_date_navigation(budget['windows_used'])
                        continue
                    if self._reserve_pending_readback(str(route_error)):
                        continue
                    raise route_error
                if decision.outcome == Outcome.STOP:
                    if self._reserve_pending_readback(decision.reason):
                        continue
                    reason = decision.reason
                    break
                if decision.reason == 'hif_cultivation_completed_with_memory':
                    closing = (after['handoff'] or {}).get('closing') or {}
                    page = host._page
                    terminal = (after['handoff'] or {}).get('terminal_receipt') or {}
                    generation = closing.get('generation') or {}
                    home_frames = closing.get('home', {}).get('frame_ids', ())
                    if (closing.get('state') != 'complete' or not has_single_frame_id(home_frames)
                            or generation.get('status') != 'complete' or not has_single_frame_id(generation.get('receipt_frame_ids'))
                            or terminal.get('round_number') not in (1, 2)
                            or closing.get('round_number') != terminal.get('round_number') or self.terminal_report is None
                            or self.terminal_report.get('terminal_settlement') != terminal
                            or page is None or page.observation.scene != 'ending_home'
                            or page.frame_ids != tuple(home_frames) or validate_observation(page.observation) is not None
                            or set(home_frames).intersection((*generation['receipt_frame_ids'], *terminal.get('frame_ids', ())))
                            or after['transition']['pending'] is not None or after['ledger']['pending'] is not None):
                        raise ValueError('automatic completion lacks its actual memory and home receipt')
                    self._checkpoint(reason=decision.reason, complete=True)
                    self.report.update(status='COMPLETE', reason=decision.reason)
                    return decision
                if decision.reason == 'battle_requires_task275':
                    request = self._handoff(self._size(), budget['windows_used'], battle_policy=config.battle_policy)
                    if self.pause_at_battle:
                        reason = 'hif_battle_handoff_ready_for_task275'
                        break
                    decision = self._battle(request)
                    if decision is not None:
                        reason = decision.reason
                        break
                elif decision.reason not in CONTINUE_BOUNDARIES:
                    reason = decision.reason
                    break
                elif not attempted and _progress(record) == _progress(after):
                    reason = 'automatic_nonbattle_boundary_has_no_confirmed_progress'
                    break
        except Exception as error:
            reason = str(error) or type(error).__name__
            self.report.update(error=reason, error_type=type(error).__name__)
        self.report.update(status='STOP', reason=reason)
        try:
            self._checkpoint(reason=reason)
        except Exception as error:
            self.report['checkpoint_error'] = str(error)
        return Decision(Outcome.STOP, reason)
