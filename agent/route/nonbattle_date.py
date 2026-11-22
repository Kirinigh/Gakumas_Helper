"""Same-owner date navigation without replacing the original business gate.

Both transitions and their receipts stay in the existing shared record. Date
completion means only that the original producer observed its business page;
the ordinary host still owns every cost and pending-action reconciliation.
"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from types import SimpleNamespace
from dataclasses import asdict, replace

from .hif import HifConfig, active_hif_role
from .host import HifRouteHost, HifRoutePage
from .engine import Target, Outcome, Decision, Observation, TransitionGate, source_frames_match, validate_observation
from .ownership import write_hif_record, locked_hif_record
from .checkpoint import restore_transition, transition_checkpoint
from .maa_bridge import MAX_NAVIGATION_INPUT_AGE_MS, HifRuntimeHooks, run_hif_sequence
from .date_recovery import ACTION, DateRecoveryProducer, read_date_page
from .input_receipt import persist_input_event, original_input_report

KEY = 'route_date_navigation'


class NonbattleDateObserved(ValueError):
    """A complete current known date surface; never an arbitrary STOP retry."""


class _NetworkDownloadObserved(NonbattleDateObserved):
    """The successful retry is loading assets; keep observing its original receipt."""


def _network_download(rows):
    from .nonbattle_observation import _within
    return any(row['score'] >= .85 and _within(row['box'], (0, 1160, 720, 120))
        and (row['text'] == 'データダウンロード中' or row['text'].upper().rstrip('.…') == 'NOWLOADING')
        for row in rows)


def navigation_closed(record):
    """An unknown or unresolved child cannot mint a settled pause receipt."""
    child = (record.get('handoff') or {}).get('nonbattle_context', {}).get(KEY)
    if child is None:
        return True
    if not isinstance(child, dict) or child.get('schema_version') != 1 or child.get('status') != 'complete':
        return False
    try:
        gate = restore_transition(child.get('transition'), clock_epoch=record['transition']['clock_epoch'])
    except (ValueError, TypeError, KeyError):
        return False
    navigation, proof = child.get('navigation') or {}, child.get('business_receipt') or {}
    if not isinstance(navigation, dict) or not isinstance(proof, dict):
        return False
    frames = child.get('completion_frame_ids')
    business_frames = proof.get('frame_ids')
    try:
        business = Observation(**proof['observation'])
    except (KeyError, TypeError):
        return False
    return (gate.pending is None and not gate.stopped and navigation.get('schema_version') == 2
        and navigation.get('completed') is True
        and navigation.get('retry_reservation') is None and isinstance(frames, list) and len(frames) == 1
        and isinstance(frames[0], str) and bool(frames[0])
        and isinstance(business.page_id, str) and isinstance(business_frames, list)
        and all(isinstance(key, str) and key for key in business_frames)
        and source_frames_match(business, business_frames) and frames == navigation.get('completion_frame_ids')
        and frames[0] == business_frames[-1])


def business_scope(record):
    from .cultivation import _budget_container
    handoff = deepcopy(record.get('handoff') or {})
    handoff.get('nonbattle_context', {}).pop(KEY, None)
    budget = _budget_container(handoff)
    if budget is not None: budget.pop('automatic_cultivation', None)
    return dict(run_id=record['run_id'], ownership=deepcopy(record['ownership']),
        ledger=deepcopy(record['ledger']), transition=deepcopy(record['transition']), handoff=handoff)


def business_saved_day(host):
    from . import rewards, activity_reward
    gate, ledger = host.hooks.transition_gate, host.ledger
    if gate.pending is not None and ledger.pending is not None and ledger.pending.expected.day != ledger.state.day:
        return ledger.pending.expected.day
    course, activity = rewards.request(host), activity_reward.request(host)
    if course is not None and (course['stage'] == 'census' or gate.pending is not None
            and gate.pending.target.kind == 'training_reward_ack'):
        return course['next_day']
    if activity is not None and (activity['stage'] == 'census' or gate.pending is not None
            and gate.pending.target.kind == 'activity_ack_card' and len(activity['cards']) == 2):
        return activity['census_day']
    return ledger.state.day


def detect_date(context, image, rows):
    host = getattr(context, 'gkh_hif_route_host', None)
    if (host is None or not getattr(host, '_active', False) or host.ledger is None
            or (host.ownership or {}).get('owner') != 'task600' or host.ownership.get('phase') != 'nonbattle'):
        return
    if isinstance(host, NonbattleDateNavigation) and host._submitted_retry() and _network_download(rows):
        raise _NetworkDownloadObserved('network retry is downloading game assets')
    try:
        page = read_date_page(image, rows, saved_day=business_saved_day(host), context=context, nonbattle=True)
    except ValueError:
        return
    raise NonbattleDateObserved('current complete nonbattle date surface: '+page['scene'])


class NonbattleDateNavigation:
    validate_dispatch = HifRouteHost.validate_dispatch
    _matches_prepared_business_record = HifRouteHost._matches_prepared_business_record
    _prepared_business_record = HifRouteHost._prepared_business_record

    def __init__(self, runtime, producer):
        self.runtime, self.producer = runtime, producer
        self.context, self.path, self.clock = runtime.bound, runtime.path, runtime.clock
        self.run_id, self.game_pid = runtime.run_id, runtime.game_pid
        self._active, self._page, self.input_events = False, None, []
        self.state = self.handoff = None
        self._navigation_only = False
        self._input_adapter = getattr(self.context, 'gkh_hif_capture_adapter', None)

    def _input_age_limit(self, proposal):
        return MAX_NAVIGATION_INPUT_AGE_MS

    def _submitted_retry(self):
        pending = self.hooks.transition_gate.pending
        receipt = self.handoff['nonbattle_context'].get('route_input_receipt') or {}
        return (pending is not None and pending.target.kind == 'date_network_retry'
            and receipt.get('dispatch_state') == 'succeeded')

    def _proposal_timeout_ms(self, proposal):
        original = self.producer.host.hooks.transition_gate
        if (proposal.target.kind == 'date_network_retry' and original.pending is not None
                and original.pending.target.kind == 'confirm_training'):
            # Bind the new retry's observation window before input. The
            # original submitted course keeps its own deadline and receipt.
            return original.timeout_ms
        return 15_000

    def _read(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    def _guard(self, record):
        if business_scope(record) != self.state['business_scope']:
            raise ValueError('date navigation cannot replace its original business scope, gate or receipt')
        if (self.runtime.authorise_capture() is not True or self.runtime.authorise_input() is not True
                or self.runtime.context.tasker.stopping):
            raise ValueError('same-owner date navigation permission was revoked')
        from .run_budget import validate_budget
        from .cultivation import _budget_container
        budget = (_budget_container(record.get('handoff') or {}) or {}).get('automatic_cultivation')
        validate_budget(budget, run_id=self.run_id, game_pid=self.game_pid,
            epoch=self.clock.epoch, now=self.clock.now())
        self.runtime._size()  # Recheck the actual same-PID elevated endpoint.

    def _write(self, state, *, receipt_only=False):
        record = self._read()
        if receipt_only:
            # Recording an SDK result is not another input. A rejected scope
            # may already be closed; still preserve its exact before-dispatch
            # receipt without changing the original business or child state.
            if (business_scope(record) != self.state['business_scope']
                    or record['handoff']['nonbattle_context'].get(KEY) != self.state):
                raise ValueError('date receipt cannot overwrite changed business or navigation')
        else:
            self._guard(record)
        state = json.loads(json.dumps(state, allow_nan=False))
        candidate = deepcopy(record)
        candidate['handoff'].setdefault('nonbattle_context', {})[KEY] = deepcopy(state)
        write_hif_record(self.path, candidate)
        self.state = deepcopy(state)

    def _dispatch_record(self):
        record = self._read()
        self._guard(record)
        child = record['handoff']['nonbattle_context'].get(KEY)
        if child != self.state:
            raise ValueError('date dispatch prepared navigation changed')
        context = dict(date_recovery=child['navigation'], route_input_receipt=child['route_input_receipt'])
        return dict(schema_version=3, run_id=record['run_id'], ownership=record['ownership'], ledger=record['ledger'],
            transition=child['transition'], handoff=dict(nonbattle_context=context))

    def _validate_business(self, record):
        owner = record['ownership']
        if (record['run_id'] != self.run_id or owner.get('game_pid') != self.game_pid
                or (owner.get('owner'), owner.get('phase')) != ('task600', 'nonbattle')
                or record.get('ledger') is None):
            raise ValueError('date navigation lacks its current complete nonbattle owner')
        observed = self.producer.host
        if (observed.ownership != owner or json.loads(json.dumps(observed.ledger.record())) != record['ledger']
                or json.loads(json.dumps(transition_checkpoint(observed.hooks.transition_gate, self.clock.epoch))) != record['transition']):
            raise ValueError('date navigation original business observer is stale')
        root = restore_transition(record['transition'], clock_epoch=self.clock.epoch)
        if root.pending is not None:
            carrier = SimpleNamespace(handoff=record['handoff'], run_id=self.run_id, game_pid=self.game_pid, clock=self.clock)
            original_input_report(carrier, root)
        elif record['ledger'].get('pending') is not None:
            carrier = (record.get('handoff') or {}).get('nonbattle_context', {}).get('route_input_receipt') or {}
            original = restore_transition(carrier.get('original_transition'), clock_epoch=self.clock.epoch)
            report = original_input_report(SimpleNamespace(handoff=record['handoff'], run_id=self.run_id,
                game_pid=self.game_pid, clock=self.clock), original)
            self.runtime._confirmed_reward_reservation(record, original, report)

    def _initialise(self, record):
        self._validate_business(record)
        saved = (record.get('handoff') or {}).get('nonbattle_context', {}).get(KEY)
        if saved is not None and (not isinstance(saved, dict) or saved.get('status') not in ('active', 'complete')
                or saved.get('schema_version') != 1 or type(saved.get('episode')) is not int or saved['episode'] < 1):
            raise ValueError('same-owner date navigation metadata is unknown')
        if saved is not None and saved['status'] == 'complete' and not navigation_closed(record):
            raise ValueError('previous same-owner date navigation is not independently complete')
        scope = business_scope(record)
        if saved is not None and saved.get('status') == 'active':
            if saved.get('schema_version') != 1 or saved.get('business_scope') != scope:
                raise ValueError('active date child belongs to another business scope')
            self.state = deepcopy(saved)
        else:
            episode = (saved or {}).get('episode', 0)+1
            self.state = dict(schema_version=1, status='active', business_scope=scope, episode=episode,
                source_id=f'{self.clock.now()}-route-date', transition=transition_checkpoint(TransitionGate(), self.clock.epoch),
                previous_episodes=[*(saved or {}).get('previous_episodes', ()), *([{
                    key: deepcopy(value) for key, value in saved.items() if key != 'previous_episodes'}] if saved else [])])
        self.ownership, self.ledger = record['ownership'], self.producer.host.ledger
        self.saved_day = business_saved_day(self.producer.host)
        self.handoff = dict(nonbattle_context={})
        if self.state.get('navigation') is not None:
            self.handoff['nonbattle_context']['date_recovery'] = deepcopy(self.state['navigation'])
        if self.state.get('route_input_receipt') is not None:
            self.handoff['nonbattle_context']['route_input_receipt'] = deepcopy(self.state['route_input_receipt'])
        gate = restore_transition(self.state['transition'], clock_epoch=self.clock.epoch)
        self.hooks = HifRuntimeHooks(self.observe, lambda: self._allowed(), lambda: self._active and self._allowed(),
            transition_gate=gate, save_transition=self._save, clock_epoch=self.clock.epoch, clock=self.clock.now,
            cancel_unissued=self._cancel, decide=self.decide, input_event=self._event, input_result=self._result,
            max_input_age_ms=MAX_NAVIGATION_INPUT_AGE_MS, proposal_timeout_ms=self._proposal_timeout_ms)
        self.date = DateRecoveryProducer(self.context, run_id=self.run_id, game_pid=self.game_pid,
            handoff_id=f'route-owner-{self.ownership["revision"]}', source_report=dict(id=self.state['source_id']), clock=self.clock)
        self.date.host = self
        from .production import HifRecoveryProducer
        self.frames = HifRecoveryProducer(self.context, run_id=self.run_id, game_pid=self.game_pid, clock=self.clock,
            evidence_sink=self.runtime.evidence_sink)
        self.frames.host = self
        self._input_source = self.frames.freeze_input_source
        self._save(self.state['transition'])

    def _allowed(self):
        self._guard(self._read())
        return True

    def _save(self, checkpoint, *, cancel_unissued=False):
        gate = restore_transition(checkpoint, clock_epoch=self.clock.epoch)
        metadata = self.date.checkpoint(gate, cancel_unissued=cancel_unissued)
        if self._page is not None and self._page.observation.scene == 'saved_nonbattle' and gate.pending is None:
            metadata.update(completed=True, completion_frame_ids=list(self._page.frame_ids))
        state = dict(self.state, transition=deepcopy(checkpoint), navigation=metadata)
        self._write(state)
        self.handoff['nonbattle_context']['date_recovery'] = deepcopy(metadata)
        return True

    def _cancel(self):
        return self._save(transition_checkpoint(TransitionGate(), self.clock.epoch), cancel_unissued=True)

    def _receipt(self, envelope):
        state = dict(self.state, transition=deepcopy(envelope['transition']),
            route_input_receipt=deepcopy(envelope['handoff']['nonbattle_context']['route_input_receipt']))
        self._write(state, receipt_only=True)

    def _event(self, event):
        self.input_events.append(event)
        persist_input_event(self, event, write_record=self._receipt)

    def _result(self, event):
        persist_input_event(self, event, complete=True, write_record=self._receipt)

    def observe(self, *, reconciliation=False):
        from .production import _current_rows
        started_observing = self.clock.now()
        deadline = started_observing + 8000
        first_download = True
        gate = self.hooks.transition_gate
        recovering_login = not (self.state.get('navigation') or {}).get('resume_requested')
        if gate.pending is not None and not reconciliation:
            deadline = min(deadline, gate.issued_at_ms+gate.timeout_ms)
        while True:
            self._allowed()
            started = self.clock.now()
            if started >= deadline:
                raise ValueError('same-owner date navigation capture deadline reached')
            job = self.context.tasker.controller.post_screencap().wait()
            if not job.succeeded: raise ValueError('same-owner date navigation capture failed')
            image = job.get()
            rows = _current_rows(self.context, image, date_probe=False)
            if self._submitted_retry() and _network_download(rows):
                if not reconciliation:
                    # Switch to the successful receipt's read-only recovery.
                    # The issued retry and its timeout stay exactly as recorded.
                    raise _NetworkDownloadObserved('network retry is downloading game assets')
                if first_download:
                    self.frames.evidence_sink(image, dict(frame_id=f'route-frame-{self.game_pid}-{started}',
                        phase='network_asset_download', rows=rows))
                    first_download = False
                time.sleep(1)
                started_observing = self.clock.now()
                deadline = started_observing + 8000
                continue
            try:
                page = read_date_page(image, rows, saved_day=self.saved_day, context=self.context,
                    nonbattle=True, recovering_login=recovering_login)
            except ValueError:
                original = restore_transition(self.state['business_scope']['transition'], clock_epoch=self.clock.epoch)
                from .exchange import exchange_offer_next
                from .exchange_census import DateExchangeCensus, exchange_pending
                census = None
                try:
                    if (exchange_pending(self.ledger, original)
                            and (self.state.get('exchange_census') is not None or exchange_offer_next(rows) is not None)):
                        child = DateExchangeCensus(self)
                        try:
                            proof, census = child.run()
                        finally:
                            self.input_events.extend(child.input_events)
                        proof = replace(proof, observation=replace(proof.observation, scene='card_choice',
                            day=self.saved_day, idol=active_hif_role(self.producer.host)[0],
                            plan=active_hif_role(self.producer.host)[1], targets=()))
                    else:
                        proof = self.producer.observe(reconciliation=True)
                except _NetworkDownloadObserved:
                    if not reconciliation:
                        raise
                    deadline = self.clock.now() + 8000
                    continue
                except NonbattleDateObserved:
                    # A loading frame can precede the next known navigation
                    # page. Re-read that page in this same bounded batch;
                    # it is not a returned business receipt or another click.
                    continue
                seen = proof.observation
                if (validate_observation(seen) is not None or seen.run_id != self.run_id
                        or seen.day != self.saved_day
                        or (seen.idol, seen.plan, seen.scenario) != (*active_hif_role(self.producer.host), 'HIF')
                        or not source_frames_match(seen, proof.frame_ids)):
                    raise ValueError('date return lacks a fresh original-business observation')
                if original.pending is not None and seen.scene not in original.pending.expected_scenes:
                    from .rewards import same_course_animation
                    if (gate.pending is not None and gate.pending.target.kind == 'date_network_retry'
                            and same_course_animation(self.producer.host, original.pending, proof)):
                        deadline = (started_observing if reconciliation else gate.issued_at_ms) + gate.timeout_ms
                        continue
                    if census is None or census['classification'] != 'before' or seen.scene != 'card_choice':
                        raise ValueError('date return lacks a fresh original-business observation')
                self.business_receipt = dict(observation=asdict(seen), frame_ids=list(proof.frame_ids),
                    resource_receipt=asdict(proof.receipt) if proof.receipt is not None else None)
                if census is not None:
                    self.business_receipt['exchange_census'] = deepcopy(census)
                result = HifRoutePage(replace(seen, scene='saved_nonbattle',
                    page_id=f'route-date:{self.state["source_id"]}:saved_nonbattle', idol='unverified', plan='unverified',
                    targets=()), {}, frame_ids=proof.frame_ids)
                self._page = result
                return result.observation, result.boxes
            key = f'route-frame-{self.game_pid}-{started}'
            self.frames.evidence_sink(image, dict(frame_id=key, phase=page['scene'], observation=page, rows=rows))
            break
        kind = ACTION[page['scene']]
        revision = self.clock.now()
        page_id = f'route-date:{self.state["source_id"]}:{page["scene"]}'
        if page.get('page_kind'):
            page_id += ':' + page['page_kind']
        observation = Observation(self.run_id, revision, page['scene'], page_id,
            self.ledger.state.day, 'unverified', 'unverified', revision-started, 1, 'HIF', (Target(kind, kind, available=True),))
        self._page = HifRoutePage(observation, {kind: tuple(page['target'])}, frame_ids=(key,))
        return observation, self._page.boxes

    def decide(self, observation, config):
        if observation.scene == 'saved_nonbattle':
            state = deepcopy(self.state)
            state.update(status='complete', completion_frame_ids=list(self._page.frame_ids), business_receipt=self.business_receipt)
            state['navigation'] = dict(state['navigation'], completed=True, completion_frame_ids=list(self._page.frame_ids))
            self._write(state)
            return Decision(Outcome.PAUSE, 'nonbattle_date_navigation_complete_recheck_original_business')
        decision = self.date.decide(observation, config)
        if decision.proposal is not None:
            destinations = tuple('saved_nonbattle' if scene == 'saved_live' else scene
                for scene in decision.proposal.expected_scenes)
            if decision.proposal.target.kind == 'date_network_retry' and 'saved_nonbattle' not in destinations:
                destinations += ('saved_nonbattle',)
            decision = replace(decision, proposal=replace(decision.proposal,
                expected_scenes=destinations))
        return decision

    def _prepare(self):
        gate = self.hooks.transition_gate
        if gate.pending is None: return
        page_prefix = f'route-date:{self.state["source_id"]}:{gate.pending.source.scene}'
        if gate.pending.source.page_id != page_prefix and not gate.pending.source.page_id.startswith(page_prefix + ':'):
            raise ValueError('date child pending does not belong to its original navigation source')
        carrier = (self.handoff.get('nonbattle_context') or {}).get('route_input_receipt') or {}
        if carrier.get('dispatch_state') == 'cancelled_unissued':
            from .input_receipt import cancelled_unissued_report
            report = cancelled_unissued_report(self, gate)
            observation, _ = self.observe(reconciliation=True)
            event = report['route']['input_events'][0]
            if (observation.scene != gate.pending.source.scene or observation.page_id != gate.pending.source.page_id
                    or observation.revision <= gate.pending.source.revision
                    or list(self._page.boxes.get(gate.pending.target.key, ())) != event['target']
                    or set(self._page.frame_ids).intersection(row.get('frame_id') for row in report['frames'])):
                raise ValueError('unissued date child lacks its fresh unchanged source')
            metadata = self.date._state()
            metadata['failures'].append(dict(outcome='verified_unissued_date_input_cancelled', original_report=report,
                original_transition=transition_checkpoint(gate, self.clock.epoch), receipt_frame_ids=list(self._page.frame_ids)))
            self.handoff['nonbattle_context']['date_recovery'] = metadata
            self._cancel()
            gate.pending, gate.issued_at_ms, gate.stopped = None, None, False
            return
        report = original_input_report(self, gate)
        observation, _ = self.observe(reconciliation=True)
        if (observation.scene not in gate.pending.expected_scenes or observation.state_key == gate.pending.source.state_key
                or observation.revision <= gate.pending.source.revision
                or set(self._page.frame_ids).intersection(row.get('frame_id') for row in report.get('frames', ()))):
            raise ValueError('date child return differs from its original successful navigation')
        metadata = self.date._state()
        metadata['failures'].append(dict(source_result=report['id'],
            original_transition=transition_checkpoint(gate, self.clock.epoch), receipt_frame_ids=list(self._page.frame_ids)))
        self.handoff['nonbattle_context']['date_recovery'] = metadata
        self._save(transition_checkpoint(TransitionGate(timeout_ms=gate.timeout_ms), self.clock.epoch))
        gate.pending, gate.issued_at_ms, gate.stopped = None, None, False

    def run(self, *, timeout_ms=None):
        with locked_hif_record(self.path):
            self._active = True
            original_host = self.context.gkh_hif_route_host
            try:
                self._initialise(self._read())
                self.context.gkh_hif_route_host = self
                while True:
                    self._prepare()
                    try:
                        return run_hif_sequence(self.context, HifConfig(mode='advance'), self.hooks,
                            max_actions=None, timeout_ms=timeout_ms)
                    except _NetworkDownloadObserved:
                        # Reconcile the successful retry without dispatching it again.
                        continue
            finally:
                self.context.gkh_hif_route_host = original_host
                self._active = False
