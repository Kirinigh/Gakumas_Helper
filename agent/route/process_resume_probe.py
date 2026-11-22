"""Finite same-save read-page probe, preserving the original business record."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, replace

from .hif import HifConfig, active_hif_role
from .host import windows_route_clock
from .engine import Target, Outcome, Decision, TransitionGate, propose_unique
from .ledger import restore_ledger
from .ownership import write_hif_record, locked_hif_record
from .checkpoint import restore_transition, transition_checkpoint
from .maa_bridge import HifRuntimeHooks, run_hif_sequence, route_input_action
from .input_receipt import persist_input_event, original_input_report
from .process_resume import KEY, SameSaveProof, _canonical, _without_probe, preview_same_save_process_resume

EXPECTED = {'recovery_menu': ('menu',), 'recovery_notebook': ('notebook', 'identity'),
    'recovery_identity': ('identity',), 'recovery_close': ('resume_home',),
    'recovery_inventory': ('inventory',), 'recovery_inventory_detail': ('inventory',),
    'recovery_inventory_scroll': ('inventory',), 'recovery_drink_detail': ('recovery_drink_detail',),
    'recovery_drink_close': ('resume_home',)}


class ProcessResumeProbe:
    def __init__(self, context, *, path, run_id, original_game_pid, endpoint, authorise_capture,
            approve_navigation, approved_budget_migration, evidence_sink=None, clock=None, capture_endpoint=None,
            authorise_probe_state=None, authorise_continuation=None, with_probe_record=None,
            authorise_engineering_continuation=None, authorise_final_engineering_continuation=None):
        from pathlib import Path
        if any(not callable(value) for value in (endpoint, authorise_capture, approve_navigation)):
            raise ValueError('same-save probe needs trusted native capture and per-navigation callbacks')
        if capture_endpoint is not None and not callable(capture_endpoint):
            raise ValueError('same-save combined capture endpoint must be a trusted callable')
        if authorise_probe_state is not None and not callable(authorise_probe_state):
            raise ValueError('same-save probe state permission must be a trusted callable')
        self.context, self.path, self.run_id = context, Path(path), run_id
        self.endpoint, self.authorise_capture, self.approve_navigation = endpoint, authorise_capture, approve_navigation
        self.capture_endpoint = capture_endpoint
        self.authorise_probe_state = authorise_probe_state
        if any(value is not None and not callable(value) for value in (
                authorise_continuation, authorise_engineering_continuation, authorise_final_engineering_continuation, with_probe_record)):
            raise ValueError('probe continuation and record access require trusted native callbacks')
        self.with_probe_record = with_probe_record
        self._capture_sources, self._pending_capture = {}, None
        self.original_game_pid, self.migration, self.clock = original_game_pid, deepcopy(approved_budget_migration), clock or windows_route_clock()
        self.evidence_sink, self._active, self._page, self.input_events = evidence_sink, False, None, []
        self.state = self.observer = self.ledger = None

    def _read(self):
        return json.loads(self.path.read_text(encoding='utf-8'))

    def _check_business(self, record_snapshot=None):
        if self.context.tasker.stopping:
            raise ValueError('same-save probe native task permission was revoked')
        raw = self.path.read_bytes()
        current = None if record_snapshot is None else record_snapshot.consume(raw, self.clock.epoch)
        if current is None:
            current = json.loads(raw)
        if _without_probe(current, detached=False) != self.original:
            raise ValueError('same-save probe original complete business record changed')
        if current['handoff'].get('nonbattle_context', {}).get(KEY) != self._acknowledged_probe:
            raise ValueError('same-save probe journal changed outside its acknowledged write')
        now = self.clock.now()
        if now < getattr(self, '_last_budget_at_ms', self.state['first_started_at_ms']):
            raise ValueError('same-save probe accounting clock moved backwards')
        self._last_budget_at_ms = now
        return raw, current

    def _state_record(self):
        def consume(native, snapshot):
            if native is not True:
                raise ValueError('same-save private state permission was revoked')
            return self._check_business(snapshot)
        return self.with_probe_record('state', consume)

    def _state_guard(self):
        if self.with_probe_record is not None:
            self._state_record()
            return True
        if self.authorise_probe_state is None:
            return bool(self._guard())
        if self.authorise_probe_state() is not True:
            raise ValueError('same-save probe state permission was revoked')
        self._check_business()
        return True

    def _guard(self, *, capture_endpoints=None, record_snapshot=None):
        from .process_resume import _endpoints
        if capture_endpoints is None and self.with_probe_record is not None:
            return self.with_probe_record('capture', lambda native, snapshot:
                self._guard(capture_endpoints=native, record_snapshot=snapshot))
        if capture_endpoints is None and self.capture_endpoint is None and self.authorise_capture() is not True:
            raise ValueError('same-save probe native task permission was revoked')
        self._check_business(record_snapshot)
        # Only this synchronous native operation can supply its fresh result.
        # State-only permission never grants capture, publication or dispatch.
        endpoints = capture_endpoints if capture_endpoints is not None else (
            self.capture_endpoint() if self.capture_endpoint is not None else self.endpoint())
        _endpoints(endpoints, self.game_pid)
        if _canonical(endpoints['game']) != self.game_identity:
            raise ValueError('same-save probe actual game instance or geometry changed')
        return endpoints

    def _write(self, state, *, audit_only=False):
        checked = None
        if not audit_only:
            if self.with_probe_record is not None:
                checked = self._state_record()
            else:
                self._state_guard()
        elif not self._active:
            raise ValueError('same-save probe input audit has no active original lease')
        if checked is not None:
            raw, record = checked
            if self.path.read_bytes() != raw:
                raise ValueError('same-save probe full source bytes changed before its final write CAS')
        else:
            record = self._read()
        if _without_probe(record, detached=False) != self.original:
            raise ValueError('same-save probe business snapshot changed before publication')
        if record['handoff'].get('nonbattle_context', {}).get(KEY) != self._acknowledged_probe:
            raise ValueError('same-save probe journal changed before its acknowledged write')
        record['handoff'].setdefault('nonbattle_context', {})[KEY] = state
        write_hif_record(self.path, record)
        self.state = _canonical(state)
        self._acknowledged_probe = self.state

    def _observer_state(self):
        return deepcopy(self.handoff)

    def _save(self, checkpoint, **kwargs):
        self._write(dict(self.state, transition=deepcopy(checkpoint), observer=self._observer_state()))
        return True

    def _cancel(self):
        gate = self.hooks.transition_gate
        original = transition_checkpoint(gate, self.clock.epoch)
        cancelled = [*self.state.get('cancelled_unissued', []), original]
        journal, observer = list(self.state['inputs']), self._observer_state()
        event = self.input_events[-1] if self.input_events else None
        timing = {}
        if (journal and event is not None and event.get('dispatch_attempted') is False
                and _canonical(journal[-1]['original_transition']) == _canonical(original)):
            carrier = deepcopy(journal[-1])
            carrier['dispatch_state'] = 'cancelled_unissued'
            carrier['report']['route']['input_events'] = [dict(deepcopy(event),
                rejected_before_dispatch=True, success=False, uncertain=False)]
            journal[-1] = carrier
            observer['nonbattle_context']['route_input_receipt'] = deepcopy(carrier)
            ages = {name: event[key] for name, key in (
                ('before_freeze_ms', 'source_age_before_freeze_ms'),
                ('after_freeze_ms', 'source_age_after_freeze_ms'),
                ('after_native_approval_ms', 'source_age_after_native_approval_ms')) if key in event}
            if any(value > 1000 for value in ages.values()):
                timing['source_age_stop'] = dict(frame_ids=list(self._page.frame_ids),
                    dispatch_attempted=False, **ages)
        self._write(dict(self.state, cancelled_unissued=cancelled, inputs=journal, **timing,
            transition=transition_checkpoint(TransitionGate(timeout_ms=gate.timeout_ms), self.clock.epoch),
            observer=observer), audit_only=True)
        self.handoff = observer
        return True

    def _receipt(self, envelope):
        carrier = envelope['handoff']['nonbattle_context']['route_input_receipt']
        journal = list(self.state['inputs'])
        if journal and _canonical(journal[-1]['original_transition']) == _canonical(carrier['original_transition']):
            journal[-1] = deepcopy(carrier)
        else:
            journal.append(deepcopy(carrier))
        self._write(dict(self.state, transition=deepcopy(envelope['transition']),
            observer=deepcopy(envelope['handoff']), inputs=journal), audit_only=True)

    def _event(self, event):
        self.input_events.append(event)
        captures = [self._capture_sources.get(key) for key in self._page.frame_ids]
        if all(captures):
            self._state_guard()
            event['source_capture_checks'] = deepcopy(captures)
        else:
            # Legacy/inert producers without actual capture provenance retain
            # their original full check, never fabricated capture timestamps.
            event['endpoints'] = self._guard()
        frame_id, = self._page.frame_ids
        captured_at = int(frame_id.rsplit('-', 1)[1])
        event.update(source_age_basis='latest', source_frame_span_ms=0)

        def mark_age(stage):
            now = self.clock.now()
            event['source_age_'+stage+'_ms'] = event['source_oldest_age_'+stage+'_ms'] = now-captured_at

        mark_age('before_freeze')
        if not 0 <= event['source_age_before_freeze_ms'] <= 1000:
            self._write(dict(self.state, source_age_stop=dict(frame_ids=list(self._page.frame_ids),
                before_freeze_ms=event['source_age_before_freeze_ms'], dispatch_attempted=False)), audit_only=True)
            raise ValueError('same-save probe latest current input-source frame exceeded 1000ms')
        persist_input_event(self, event, write_record=self._receipt)
        mark_age('after_freeze')
        if not 0 <= event['source_age_after_freeze_ms'] <= 1000:
            self._write(dict(self.state, source_age_stop=dict(frame_ids=list(self._page.frame_ids),
                after_freeze_ms=event['source_age_after_freeze_ms'], dispatch_attempted=False)), audit_only=True)
            raise ValueError('same-save probe source freezing exhausted its original 1000ms gate')
        intent = dict(entry='ProduceHifAdvanceClick', action=route_input_action(event['target_kind'], event['target']),
            target_kind=event['target_kind'], source_revision=event['source_revision'], issued_at_ms=event['issued_at_ms'],
            frame_ids=list(self._page.frame_ids))
        try:
            guards = (dict(snapshot_budget_guard=lambda native, snapshot:
                bool(self._guard(capture_endpoints=native, record_snapshot=snapshot)))
                if self.with_probe_record is not None else dict(budget_guard=lambda capture_endpoints=None:
                    bool(self._guard(capture_endpoints=capture_endpoints))))
            approved = self.approve_navigation(intent, **guards)
        finally:
            mark_age('after_native_approval')
        if approved is not True:
            raise ValueError('same-save probe navigation was not approved by the actual native task')

    def _result(self, event):
        persist_input_event(self, event, complete=True, write_record=self._receipt)

    def _capture_guard(self, capture_endpoints=None, *, record_snapshot=None):
        endpoints = self._guard(capture_endpoints=capture_endpoints, record_snapshot=record_snapshot)
        self._pending_capture = dict(clock_epoch=self.clock.epoch, checked_at_ms=self.clock.now(),
            endpoints=deepcopy(endpoints))
        return True

    def _capture_evidence(self, image, metadata):
        check, self._pending_capture = self._pending_capture, None
        key = metadata.get('frame_id')
        if check is not None and isinstance(key, str) and key.startswith(f'route-frame-{self.game_pid}-'):
            started = int(key.rsplit('-', 1)[1])
            if not started <= check['checked_at_ms'] <= self.clock.now():
                raise ValueError('probe capture endpoint provenance is outside its actual frame')
            self._capture_sources[key] = dict(frame_id=key, **check)
            while len(self._capture_sources) > 64:
                self._capture_sources.pop(next(iter(self._capture_sources)))
        if self.evidence_sink is not None:
            self.evidence_sink(image, metadata)

    def _initialise(self):
        import csv
        from pathlib import Path

        from utils.runtime_paths import resolve_runtime_path
        from live_decision.catalog import SkillCardFeatureCatalog

        from .production import HifRecoveryProducer
        from .process_resume import read_bound_game_pid, reviewed_stopped_budget
        record = self._read()
        self.original = _without_probe(record)
        old_pid = read_bound_game_pid(self.original, run_id=self.run_id, original_game_pid=self.original_game_pid)
        _, self.initial_budget = reviewed_stopped_budget(self.original, old_pid=old_pid, plan=self.migration,
            clock_epoch=self.clock.epoch, now_ms=self.clock.now())
        owner, ledger = record['ownership'], record['ledger']
        root = restore_transition(record['transition'], clock_epoch=self.clock.epoch)
        if ((owner['owner'], owner['phase']) != ('task600', 'nonbattle') or root.pending is not None or root.stopped
                or ledger['pending'] is not None or (ledger['state']['day'], ledger['state']['phase']) != (3, 'schedule')
                or ledger['state']['counters']):
            raise ValueError('same-save probe requires the original closed D3 ordinary record')
        endpoints = self.capture_endpoint() if self.capture_endpoint is not None else self.endpoint()
        self.game_pid, self.game_identity = endpoints['game']['pid'], _canonical(endpoints['game'])
        if self.game_pid == old_pid:
            raise ValueError('same-save probe requires its separately verified new game instance')
        state = record['handoff'].get('nonbattle_context', {}).get(KEY)
        self._acknowledged_probe = deepcopy(state)
        if state is None:
            start = self.clock.now()
            state = dict(schema_version=1, status='active', probe_id=f'{self.game_pid}-{start}-same-save',
                first_started_at_ms=start, game_identity=deepcopy(self.game_identity),
                transition=transition_checkpoint(TransitionGate(), self.clock.epoch),
                observer={}, original=deepcopy(self.original), inputs=[], failures=[])
        elif (state.get('schema_version') != 1 or state.get('status') not in ('active', 'complete')
                or state.get('original') != self.original):
            raise ValueError('same-save probe belongs to a changed original save or game instance')
        from .process_resume import _same_probe_instance, _probe_instance_identity
        previous_identity = _probe_instance_identity(state, now=self.clock.now())
        gate = restore_transition(state['transition'], clock_epoch=self.clock.epoch)
        if self._acknowledged_probe is not None and gate.pending is None:
            from .continuation import closed_probe_journal
            closed_probe_journal(state, epoch=self.clock.epoch, run_id=self.run_id)
        if not _same_probe_instance(previous_identity, self.game_identity):
            if state['status'] != 'complete' or gate.pending is not None or gate.stopped:
                raise ValueError('same-save new instance requires its closed original probe')
            state = dict(state, status='active', current_game_identity=deepcopy(self.game_identity), observer={}, proof=None,
                instance_rebinds=[*state.get('instance_rebinds', []), dict(from_game_identity=deepcopy(previous_identity),
                    to_game_identity=deepcopy(self.game_identity), observed_at_ms=self.clock.now(),
                    previous_proof=deepcopy(state.get('proof')))])
            _probe_instance_identity(state, now=self.clock.now())
        self.state = deepcopy(state)
        self._last_budget_at_ms = max([self.state['first_started_at_ms'],
            *[row['source_returned_at_ms'] for row in self.state.get('frame_batches', ())]])
        data = resolve_runtime_path(Path('assets/data'), Path('data'))
        with (data/'p_drinks.csv').open(encoding='utf-8-sig', newline='') as stream:
            drinks = {row['id'] for row in csv.DictReader(stream)}
        self.ledger = restore_ledger(ledger, card_catalog=SkillCardFeatureCatalog.load(data/'skill_cards.csv'), drink_ids=drinks)
        self._catalog, self._drink_ids = self.ledger.card_catalog, self.ledger.drink_ids
        self.ownership = dict(owner, phase='recovery', game_pid=self.game_pid)
        self.handoff = deepcopy(self.state['observer'])
        self.hooks = HifRuntimeHooks(self.observe, self._state_guard, lambda: self._active and self._state_guard(),
            transition_gate=gate, save_transition=self._save, clock_epoch=self.clock.epoch, clock=self.clock.now,
            cancel_unissued=self._cancel, decide=self.decide, input_event=self._event, input_result=self._result)
        self.observer = HifRecoveryProducer(self.context, run_id=self.run_id, game_pid=self.game_pid,
            clock=self.clock, evidence_sink=self._capture_evidence,
            capture_guard=self._capture_guard,
            capture_record_guard=self._capture_guard if self.with_probe_record is not None else None,
            resume_probe_home=True)
        self.observer.host = self
        self._input_source = self.observer.freeze_input_source
        self._original_page = self.observer._page
        self.observer._page = self._read_page
        self.context.gkh_hif_client_size = tuple(endpoints['game']['observed_client_size'])
        if any(self.state.get(field) is not None for field in
                ('continuation', 'engineering_continuation', 'final_engineering_continuation')):
            from .continuation import probe_remaining
            probe_remaining(self.initial_budget, self.state, now=self.clock.now(), epoch=self.clock.epoch)
        self._save(self.state['transition'])

    def _read_page(self, image):
        from .daily import complete_daily_resources
        from .footer import read_footer_offset, cultivation_menu_target
        from .outing import read_outing_schedule
        from .calendar import read_calendar, read_route_resources
        from .production import _current_rows, read_route_notebook, read_nonbattle_inventory, current_inventory_summary_rows
        from .recovery_inventory import retained_identity
        if getattr(self, '_final_resources_only', False):
            from .production import current_final_resource_rows
            rows, evidence = current_final_resource_rows(self.context, image, frame_id=self.observer.current_frame_id)
            self.observer.rows = rows
            resources = read_route_resources(rows)
            offset, footer = read_footer_offset(image, context=self.context)
            self.home = dict(resources, day=3, phase='schedule', footer_offset=offset)
            return dict(self.home, phase='resume_home', target=None,
                recovery_evidence=dict(evidence, footer=footer))
        rows = summary = None
        batches = (getattr(self, 'state', None) or {}).get('frame_batches', ())
        gate = getattr(getattr(self, 'hooks', None), 'transition_gate', None)
        if gate is not None and gate.pending is None and batches and batches[-1].get('scene') == 'inventory':
            try:
                rows, summary = current_inventory_summary_rows(self.context, image,
                    frame_id=self.observer.current_frame_id)
            except (ValueError, TypeError, AttributeError):
                pass
        if rows is None:
            rows = _current_rows(self.context, image, date_probe=False)
        self.observer.rows = rows
        if self.observer.identity is None:
            saved = retained_identity(self)
            if saved is not None: self.observer.identity, self.observer.identity_frames = saved
        try:
            inventory = read_nonbattle_inventory(rows)
        except ValueError:
            inventory = None
        if inventory is not None:
            # Keep the ordinary producer's existing strict inventory-first
            # order. Its covered HUD is not a notebook resource observation.
            page = self._original_page(image, rows=rows)
            if summary is not None:
                page['recovery_evidence'] = summary
            return page
        try:
            page = read_outing_schedule(rows, selected=False)
        except ValueError:
            page = None
        if page is not None:
            rows, _ = complete_daily_resources(self.context, image, rows)
            resources = read_route_resources(rows)
            if read_calendar(rows) != (3, 'outing'):
                raise ValueError('same-save probe returned to another ordinary calendar')
            self.home_menu_box = None
            offset = None
            if self.observer.identity is None:
                self.home_menu_box = cultivation_menu_target(image)
            elif (self.observer.current_inventory() or {}).get('complete') is True:
                offset, _ = read_footer_offset(image, context=self.context)
            self.home = dict(resources, day=3, phase='schedule', footer_offset=offset)
            return dict(self.home, phase='resume_home', target=self.home_menu_box)
        try:
            frame, epoch = self.observer.current_frame_id, getattr(getattr(self, 'clock', None), 'epoch', None)
            metadata = dict(frame_id=frame, epoch=epoch) if frame is not None and epoch is not None else {}
            return read_route_notebook(self.context, image, rows, **metadata)
        except ValueError:
            return self._original_page(image, rows=rows)

    def observe(self, *, reconciliation=False):
        from .production import maintain_empty_nonbattle_drinks
        from .recovery_inventory import permanent_cards
        self._state_guard()
        options = dict(reconciliation=reconciliation)
        if getattr(self, '_final_resources_only', False):
            options['_deadline'] = self._final_resource_deadline
        proof = self.observer.observe(**options)
        seen = proof.observation
        if seen.scene in ('resume_home', 'notebook', 'identity') and self.observer.last_interval != (
                self.ledger.state.stamina, self.ledger.state.max_stamina, self.ledger.state.currency):
            raise ValueError('same-save probe current resources differ from the original complete ledger')
        notebook_day = getattr(self.observer, 'last_notebook_day', None)
        if seen.scene in ('notebook','identity') and notebook_day is not None and notebook_day != self.ledger.state.day:
            raise ValueError('same-save ordinary notebook day differs from the original complete ledger')
        if self.observer.identity is not None and (self.observer.identity.get('idol'), self.observer.identity.get('plan')) != active_hif_role(self.observer.host):
            raise ValueError('same-save probe current notebook identity differs from its approved idol')
        self.observer.last_frame_ids = proof.frame_ids
        if seen.scene == 'resume_home':
            if self.observer.identity is None:
                kind, key, box = 'recovery_menu', 'recovery_menu', self.home_menu_box
            elif (self.observer.current_inventory() or {}).get('complete') is not True:
                from .production import recognize_route_inventory_button
                image = self.observer._source_frames[proof.frame_ids[0]][0]
                matches, _, _, _ = recognize_route_inventory_button(self.context, image, tuple(self.context.gkh_hif_client_size))
                if len(matches) != 1:
                    raise ValueError('same-save inventory button is not uniquely observed')
                kind, key, box = 'recovery_inventory', 'recovery_inventory', tuple(matches[0].box)
            else:
                cards = permanent_cards(self, self.observer.current_inventory())
                if self.ledger.state.drinks:
                    raise ValueError('same-save occupied drink probe is not yet bound; do not assume remaining identity')
                images = [self.observer._source_frames[frame][0] for frame in proof.frame_ids]
                slots = [maintain_empty_nonbattle_drinks(self.context, image, self.ledger.state.drinks,
                    footer_offset=self.home['footer_offset'], revision=self.ledger.state.revision) for image in images]
                if any(any(slot['state'] != 'EMPTY' for slot in frame) for frame in slots):
                    raise ValueError('same-save four drink slots differ from the original empty inventory')
                self.handoff['current_drink_slots'] = deepcopy(slots[-1])
                self.handoff['current_resource_frame_ids'] = list(proof.frame_ids)
                state = replace(self.ledger.state, revision=seen.revision, cards=cards,
                    currency=self.home['currency'], stamina=self.home['stamina'], max_stamina=self.home['max_stamina'])
                self.proof = SameSaveProof(self.run_id, self.game_pid, self.state['probe_id'],
                    self.state['first_started_at_ms'], seen.revision, (self.observer.identity['idol'],
                    self.observer.identity['plan'], 'HIF'), state, tuple(self.observer.identity_frames),
                    tuple(self.observer.current_inventory()['frame_ids']), tuple(proof.frame_ids))
                kind = key = box = None
            seen = replace(seen, targets=(Target(key, kind, available=True),) if kind else ())
            proof = replace(proof, observation=seen, boxes={key: box} if kind else {})
        seen = replace(proof.observation, page_id=f'resume-probe:{self.state["probe_id"]}:{proof.observation.page_id}')
        proof = replace(proof, observation=seen)
        self._page = proof
        ticks = [int(frame.rsplit('-', 1)[1]) for frame in proof.frame_ids]
        timing = dict(scene=seen.scene, frame_ids=list(proof.frame_ids), first_captured_at_ms=min(ticks),
            last_captured_at_ms=max(ticks), source_returned_at_ms=self.clock.now())
        timing['oldest_age_ms'] = timing['source_returned_at_ms']-timing['first_captured_at_ms']
        timing['latest_age_ms'] = timing['source_returned_at_ms']-timing['last_captured_at_ms']
        timing['frame_span_ms'] = timing['last_captured_at_ms']-timing['first_captured_at_ms']
        self._write(dict(self.state, frame_batches=[*self.state.get('frame_batches', [])[-127:], timing]), audit_only=True)
        return proof.observation, proof.boxes

    def decide(self, observation, config):
        if observation.scene == 'resume_home' and not observation.targets:
            self._write(dict(self.state, status='complete', proof=asdict(self.proof), observer=self._observer_state()))
            return Decision(Outcome.PAUSE, 'same_save_process_probe_complete')
        targets = observation.targets
        if len(targets) != 1 or targets[0].kind not in EXPECTED:
            return Decision(Outcome.STOP, 'same_save_probe_unknown_or_non_navigation_page')
        if targets[0].kind in ('recovery_menu','recovery_notebook','recovery_identity') and any(
                carrier.get('dispatch_state') == 'succeeded' and any(
                    event.get('target_kind') == targets[0].kind and event.get('dispatch_attempted') is True
                    and event.get('success') is True and event.get('uncertain') is False
                    for event in carrier.get('report', {}).get('route', {}).get('input_events', ()))
                and all(str(row.get('frame_id', '')).startswith(f'route-frame-{self.game_pid}-')
                    for row in carrier.get('report', {}).get('frames', ()))
                for carrier in self.state['inputs']):
            name = targets[0].kind.removeprefix('recovery_')
            return Decision(Outcome.STOP, f'same_save_successful_{name}_navigation_must_not_be_reissued')
        expected = EXPECTED[targets[0].kind]
        if observation.scene == 'inventory' and targets[0].kind == 'recovery_close':
            if (self.observer.current_inventory() or {}).get('complete') is not True:
                return Decision(Outcome.STOP, 'same_save_probe_permanent_census_incomplete')
        return propose_unique(observation, targets, 'same-save-probe:'+targets[0].kind, expected)

    def _prepare(self):
        gate = self.hooks.transition_gate
        if gate.pending is None: return

        from .process_resume import _frames
        if (gate.pending.target.kind not in EXPECTED
                or gate.pending.expected_scenes != EXPECTED[gate.pending.target.kind]):
            raise ValueError('same-save probe cannot reconcile an unknown or changed navigation contract')
        report = original_input_report(self, gate)
        rows = report.get('frames', ())
        keys = tuple(row.get('frame_id') for row in rows)
        _frames(keys, game_pid=self.game_pid, started=self.state['first_started_at_ms'],
            observed=gate.pending.source.revision, count=1)
        events = report.get('route', {}).get('events', ())
        if (len(events) != 1 or events[0].get('phase') != gate.pending.source.scene
                or events[0].get('frame_ids') != list(keys)
                or events[0].get('capture_age_ms') != gate.pending.source.revision-max(int(key.rsplit('-',1)[1]) for key in keys)
                or any(row.get('phase') != gate.pending.source.scene for row in rows)):
            raise ValueError('same-save pending navigation lacks its exact registered original source')
        for row in rows:
            self.observer.original_route_image_path(row)
        self.observer.recovery_report = report
        observation, _ = self.observe(reconciliation=True)
        if (not gate.pending.source.page_id.startswith(f'resume-probe:{self.state["probe_id"]}:')
                or observation.scene not in gate.pending.expected_scenes
                or observation.state_key == gate.pending.source.state_key
                or set(self._page.frame_ids).intersection(row.get('frame_id') for row in report['frames'])):
            raise ValueError('same-save probe lacks its independently confirmed original navigation destination')
        from .engine import TransitionGate
        confirmed_at = self.clock.now()
        failures = [*self.state['failures'], dict(original_transition=transition_checkpoint(gate, self.clock.epoch),
            source_result=report['id'], destination_frame_ids=list(self._page.frame_ids),
            destination_observed_at_ms=confirmed_at, original_deadline_ms=gate.issued_at_ms+gate.timeout_ms,
            late_destination_observed=confirmed_at>=gate.issued_at_ms+gate.timeout_ms)]
        self.state = dict(self.state, failures=failures)
        self._save(transition_checkpoint(TransitionGate(timeout_ms=gate.timeout_ms), self.clock.epoch))
        gate.pending, gate.issued_at_ms, gate.stopped = None, None, False

    def run(self, *, publish=False):
        with locked_hif_record(self.path):
            self._active = True
            try:
                self._initialise()
                self._prepare()
                decision = run_hif_sequence(self.context, HifConfig(mode='advance'), self.hooks, max_actions=None, timeout_ms=None)
                if decision.reason != 'same_save_process_probe_complete':
                    raise ValueError(decision.reason)
                self._refresh_final_resources()
                endpoints = self._guard()
                result = preview_same_save_process_resume(self._read(), self.proof, endpoints=endpoints,
                    now_ms=self.clock.now(), clock_epoch=self.clock.epoch, original_game_pid=self.original_game_pid,
                    approved_budget_migration=self.migration)
                if not publish:
                    return result
                self._guard()
                from .process_resume import _fresh_resource_batch
                if self._read() != result['expected_host_record']:
                    raise ValueError('same-save original record or final current evidence changed before binding')
                _fresh_resource_batch(self.proof, now_ms=self.clock.now())
                write_hif_record(self.path, result['candidate'])
                return dict(changed=True, game_pid=result['game_pid'], binding_revision=result['binding_revision'],
                    probe_id=self.proof.probe_id, remaining_ms=result['remaining_ms'])
            finally:
                self._active = False

    def _refresh_final_resources(self):
        """One no-input refresh after full scene/census preparation, in this call."""
        if (getattr(self, '_final_resource_refresh_used', False) or self.proof is None
                or self.hooks.transition_gate.pending is not None
                or (self.observer.current_inventory() or {}).get('complete') is not True):
            raise ValueError('final resource refresh lacks its complete prepared proof')
        self._final_resource_refresh_used = True
        before = self.proof
        self._final_resource_deadline = self.clock.now() + 15_000
        self._final_resources_only = True
        try:
            observation, _ = self.observe(reconciliation=True)
        finally:
            self._final_resources_only = False
        current = self.proof
        if (observation.scene != 'resume_home'
                or replace(before.state, revision=current.state.revision) != current.state
                or replace(before, observed_at_ms=current.observed_at_ms, state=current.state,
                    resource_frame_ids=current.resource_frame_ids) != current
                or set(before.resource_frame_ids) & set(current.resource_frame_ids)
                or min(int(key.rsplit('-', 1)[1]) for key in current.resource_frame_ids)
                    <= max(int(key.rsplit('-', 1)[1]) for key in before.resource_frame_ids)):
            raise ValueError('final resource refresh changed its same-save proof or reused old frames')
        if self.decide(observation, HifConfig(mode='advance')).reason != 'same_save_process_probe_complete':
            raise ValueError('final resource refresh did not confirm the prepared same-save state')
