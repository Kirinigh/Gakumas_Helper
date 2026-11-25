"""Continuous Live invocation inside the existing Maa route host.

Uses the original observer, fast policy, choice transaction and computation
sidecars. It never launches Maa, reads command.json or acquires a second lease.
"""
from __future__ import annotations

import os
import re
import json
import time
from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
from dataclasses import asdict

from route.battle_policy import FORMAL_PROFILES, UINET21_PROFILE, NETWORK_PROFILES

from .exam_loop import ExamCycle, ExamOutcome, run_exam_session
from .hif_handoff import require_hif_handoff, bind_hif_battle_handoff, return_hif_battle_handoff
from .hif_entry_round import UNREAD_INVENTORY_SOURCES
from .hif_upstream_ui import HIF_SKIP_RECOGNITION
from .hif_live_recovery import BattleProgress, automatic_battle_run, automatic_host_active
from .hif_capture_adapter import _Forwarder, _TaskerProxy, _unsupported_pipeline
from .hif_runtime_sources import with_hif_runtime_sources
from .hif_recovery_reports import recovery_report_scope


class HifBattleContext(_Forwarder):
    """Battle-local state over the same Maa context; no second input endpoint."""

    @property
    def tasker(self):
        local = self.__dict__.get('_hif_battle_tasker')
        return local if local is not None else self._hif_wrapped.tasker

    def __getattr__(self, name):
        # Share immutable popup computation and the explicit research request.
        # The latter is revalidated against this battle's current session;
        # no game state, captured frame or input authority crosses this view.
        if name.startswith('gkh_hif_') and name not in ('gkh_hif_popup_mirror', 'gkh_hif_opening_recompute'):
            raise AttributeError(name)
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name.startswith('gkh_hif_'):
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name.startswith('gkh_hif_'):
            object.__delattr__(self, name)
        else:
            super().__delattr__(name)


class _AuditedController(_Forwarder):
    def __init__(self, controller, host):
        super().__init__(controller)
        self._hif_host = host

    def post_click(self, x, y):
        host = self._hif_host
        captured_at, deadline = host._navigation_timing()
        return host._click(x, y, before_dispatch=lambda: host._navigation_guard(deadline),
            input_deadline=deadline, authorization_floor=captured_at,
            preparation_key='native_navigation_preparation')

    def post_swipe(self, x1, y1, x2, y2, duration):
        host = self._hif_host
        _, deadline = host._navigation_timing()
        endpoints = host._endpoints(self.info)
        host._retain_current_frame('pre-swipe')
        host.result['native_navigation_preparation'] = dict(kind='swipe', start=[x1, y1],
            end=[x2, y2], duration=duration, endpoints=endpoints, dispatch_state='prepared_unconfirmed')
        host._persist()
        host._navigation_guard(deadline)
        host.result['native_input_dispatches'].append(dict(kind='swipe', start=[x1, y1],
            end=[x2, y2], duration=duration, endpoints=endpoints))
        host.result['input_count'] = len(host.result['native_input_dispatches'])
        return self._hif_wrapped.post_swipe(x1, y1, x2, y2, duration)


class _AuditedContext(_Forwarder):
    def __init__(self, context, host):
        super().__init__(context)
        self._hif_host = host
        self._hif_tasker = _TaskerProxy(context.tasker, _AuditedController(context.tasker.controller, host))

    @property
    def tasker(self):
        return self._hif_tasker

    # All existing Live observers use the audited controller Click/Swipe path.
    # Never forward a pipeline that would bypass this host's dispatch evidence.
    run_action = staticmethod(_unsupported_pipeline)

    def run_recognition(self, node, image, **kwargs):
        started = time.perf_counter()
        skip = 'ProduceRecognitionSkipRound'
        if node in (skip, 'Task275HifSceneAnd', 'Task275HifSceneOr'):
            override = kwargs.get('pipeline_override') or {}
            kwargs['pipeline_override'] = {**override, skip: {**HIF_SKIP_RECOGNITION, **override.get(skip, {})}}
        hit = self._hif_wrapped.run_recognition(node, image, **kwargs)
        host = self._hif_host
        host.result.setdefault('recognition_timing_ms', {}).setdefault(node, []).append(
            (time.perf_counter() - started) * 1000)
        if (node in ('Task200Visualstamina', 'ProduceRecognitionCards', 'ProduceLiveInventoryButton', 'HifRoundResult')
                or node.startswith(('Task275HifSelect', 'LiveInventory', 'LiveStatus', 'HifHistory'))):
            host.result['live_raw_fields'].append(dict(node=node, raw=hit.raw_detail if hit else None))
        if node in ('HifRoundResult', 'ProduceRecognitionCards', 'HifHistoryLines'):
            host._retain_frame(image, node)
        return hit


class HifLiveHost:
    """One explicitly selected battle policy with bounded observer recovery."""

    def __init__(self, context, session, *, runtime, attempt_id, params, card_catalog, drink_ids,
                 observer_module=None, host_attempt=None):
        if not isinstance(attempt_id, str) or re.fullmatch(r'[0-9]+-[a-z0-9-]+', attempt_id) is None:
            raise ValueError('HIF Live attempt must have a unique bounded record name')
        if observer_module is None:
            from custom.action import live_decision as observer_module
        self.module = observer_module
        self.params = observer_module._validate_runtime_params(dict(params))
        if (self.params['hif_policy_profile'] not in FORMAL_PROFILES
                or self.params['live_decision_mode'] != 'opening_execute'):
            raise ValueError('continuous host requires an explicitly authorised battle profile')
        self.context = bind_hif_battle_handoff(context, session,
            card_catalog=card_catalog, drink_ids=drink_ids)
        require_hif_handoff(self.context, self.params)
        self.controller = self.context.tasker.controller
        self.context = _AuditedContext(self.context, self)
        from utils.local_data import get_runtime_store

        self.capture_store = get_runtime_store()
        self.runtime, self.attempt_id = Path(runtime), attempt_id
        self.host_attempt = host_attempt or attempt_id
        self.records = []
        self.result = None
        self._used = False
        self._completed_inputs = 0
        self._frames = {}
        self._sealed_terminal_source = None
        self.automatic_control = None
        self._control_json_cache = None
        self._automatic_previous_record = None
        self._parent_deadline = None
        self._last_parent_check = None
        self._navigation_resume = None
        self._opening_resume = None
        self._opening_source = None
        self._battle_progress = None

    def _endpoints(self, info, **kwargs):
        from utils.maa_input import verify_maa_input_endpoints

        endpoints = verify_maa_input_endpoints(info, expected_client_size=self.context.gkh_hif_client_size)
        if endpoints['game']['pid'] != self.context.gkh_hif_battle_handoff.game_pid:
            raise ValueError('HIF input game process differs from its handoff')
        return endpoints

    def _capture_deadline(self, captured_at=None):
        if captured_at is None:
            captured_at = self.controller.frame_metadata(self.controller.cached_image)['captured_at']
        return captured_at + self.params['max_snapshot_age_ms'] / 1000

    def _navigation_timing(self, captured_at=None):
        if captured_at is None:
            captured_at = self.controller.frame_metadata(self.controller.cached_image)['captured_at']
        if getattr(self.context, 'gkh_hif_upstream_card_ui', False) is True:
            return captured_at, None
        return captured_at, self._capture_deadline(captured_at)

    def _navigation_guard(self, deadline):
        require_hif_handoff(self.context, self.params)
        self._require_time()
        checked_at = time.perf_counter()
        if deadline is not None and checked_at > deadline:
            raise ValueError('HIF navigation evidence expired before Maa dispatch')
        return checked_at

    def _require_time(self):
        if self.automatic_control is not None:
            from .hif_live_recovery import require_deadline
            require_deadline(self.automatic_control)
        elif not automatic_host_active() and self._parent_deadline is not None:
            now = time.time()
            if not self._last_parent_check <= now < self._parent_deadline:
                raise ValueError('HIF parent deadline exhausted or clock moved backwards')
            self._last_parent_check = now

    def _retain_current_frame(self, label):
        self._retain_frame(self.controller.cached_image, label)

    def _retain_frame(self, image, label):
        frame_id = self.module.frame_identifier(image)
        kind = 'failure_scene' if label.endswith('-failure') else 'screenshot'
        mode = self.capture_store.config.mode
        failure_only = mode in ('failure', 'roi') and kind == 'failure_scene'
        retain = mode == 'screenshot' or failure_only
        key = (frame_id, kind) if failure_only else frame_id
        if key in self._frames:
            return
        if len(self._frames) >= 128:
            key_old = next(iter(self._frames))
            record, payload, _ = self._frames.pop(key_old)
            if payload is not None:
                record['image_not_retained'] = 'bounded transaction diagnostic eviction'
        record = dict(frame_id=frame_id, label=label)
        self.result['frames'].append(record)
        self._frames[key] = record, image.copy() if retain else None, kind

    def _flush_frames(self):
        frames = tuple((record['label'], image, record, kind) for record, image, kind in self._frames.values()
                       if image is not None)
        self._frames.clear()
        if not frames:
            return
        import cv2

        from .hif_live_io import persist_deferred_frames

        store = self.capture_store
        session = store.start_session('task275-live-handoff')
        reports = []
        for kind in ('screenshot', 'failure_scene'):
            batch = [(label, image, record) for label, image, record, capture in frames if capture == kind]
            if batch:
                reports.append(persist_deferred_frames(store, session, batch, attempt_id=self.result['id'],
                    encode=cv2.imencode, capture_kind=kind))
        self.result.setdefault('frame_persistence_batches', []).extend(reports)
        if any(report['failed_count'] for report in reports):
            raise ValueError('HIF diagnostic frame batch was not completely retained')

    def _encode(self, value):
        if callable(getattr(value, 'to_dict', None)):
            return value.to_dict()
        return self.module._live_event_json_default(value)

    def _encode_record(self):
        """Reuse this record's immutable history encoding, never an input grant.

        The initial record is persisted before any evidence capture. Its large
        history is encoded there once and copied into read-only containers.
        Later writes require those same containers; nested edits fail at their
        source, and all current fields are encoded anew. Disk JSON is unchanged.
        """
        from .hif_live_io import freeze_record_history

        def encode(value):
            return json.dumps(value, ensure_ascii=False, separators=(',', ':'),
                allow_nan=False, default=self._encode).encode('utf-8')

        control = self.automatic_control
        if control is None:
            return (encode(self.result),)
        history_fields = {'engineering_window', 'engineering_repair',
                          'opening_parent_window', 'engineering_followups'}
        fields = history_fields.intersection(control)
        cache = getattr(self, '_control_json_cache', None)
        if cache is None:
            fragments = {}
            for key in fields:
                text = encode(control[key])
                fragments[key] = (freeze_record_history(control[key]), text)
            for key, (frozen, _) in fragments.items():
                control[key] = self.result['automatic_battle'][key] = frozen
            self._control_json_cache = cache = (control, fragments)
        original, fragments = cache
        if control is not original or fields != fragments.keys():
            raise ValueError('HIF durable history source changed inside the record')
        parts = [b'{']
        for key, value in self.result['automatic_battle'].items():
            if key in fragments:
                frozen, text = fragments[key]
                if value is not frozen:
                    raise ValueError('HIF durable history value changed inside the record')
            else:
                text = encode(value)
            if len(parts) > 1:
                parts.append(b',')
            parts.extend((encode(key), b':', text))
        parts.append(b'}')
        encoded = [b'{']
        for key, value in self.result.items():
            if len(encoded) > 1:
                encoded.append(b',')
            encoded.extend((encode(key), b':'))
            encoded.extend(parts if key == 'automatic_battle' else (encode(value),))
        encoded.append(b'}')
        # Reuse only the already-frozen history's encoded bytes. Joining and
        # re-encoding a 63 MB record at every click costs more than its I/O.
        # All current fields are freshly encoded; the disk JSON is identical.
        return encoded

    def _persist(self):
        # Once publication can reference this complete terminal report, later
        # unwinding must not replace its evidence with an exception or STOP.
        if self._sealed_terminal_source is not None:
            return
        if self.result is None:
            raise ValueError('HIF Live transaction has no durable record')
        started = time.perf_counter()
        context = self.context
        if self.automatic_control is not None:
            # Only the top-level clock/counters change during an invocation.
            # Window appenders build new deep-copied controls; their existing
            # nested history is read-only, so snapshot the mutable root only.
            self.result['automatic_battle'] = dict(self.automatic_control)
        self.result['total_input_count'] = self._completed_inputs + self.result['input_count']
        for field, attribute in (('observed_live_context', 'gkh_hif_live_context'),
                ('round_context', 'gkh_hif_round_context')):
            value = getattr(context, attribute, None)
            if value is not None:
                self.result[field] = asdict(value)
        self.result['handoff_opening'] = getattr(context, 'gkh_hif_handoff_opening', None)
        temporary = self.output.with_suffix('.tmp')
        snapshot_finished = time.perf_counter()
        encoded = self._encode_record()
        encoded_at = time.perf_counter()
        with temporary.open('wb') as stream:
            stream.writelines(encoded)
            stream.flush()
            written_at = time.perf_counter()
            os.fsync(stream.fileno())
        # A Windows reader can briefly deny deletion of the old record. Retry
        # only publication of these already-flushed bytes, never the action.
        # The original input deadline is still checked after persistence.
        for delay in (0.02, 0.05, 0.1, None):
            try:
                temporary.replace(self.output)
                break
            except PermissionError:
                if delay is None:
                    raise
                time.sleep(delay)
        persisted_at = time.perf_counter()
        # Returned to the existing input receipt after this write; no extra I/O.
        return dict(snapshot_ms=(snapshot_finished - started) * 1000,
            encode_ms=(encoded_at - snapshot_finished) * 1000,
            write_flush_ms=(written_at - encoded_at) * 1000,
            fsync_replace_ms=(persisted_at - written_at) * 1000)

    def _restore_opening_drink_panel(self):
        """Use the existing cancel-only reader, then require unchanged Live."""
        from .opening_observer import opening_component_params
        from .hif_navigation_recovery import verify_unstarted_core

        params = opening_component_params(self.params, 'drink_inventory_observe')
        params['live_decision_mode'] = 'drink_reset'
        start = len(self.result.get('published_components', ()))
        self.module.ProduceLiveDecisionObserve().run(self.context,
            SimpleNamespace(custom_action_param=json.dumps(params)))
        reports = self.result.get('published_components', ())[start:]
        if (len(reports) != 1 or reports[0].get('accepted') is not True
                or reports[0].get('executed') is not False or reports[0].get('execution_submit_count') != 0):
            raise ValueError('HIF opening drink panel was not restored')
        events = reports[0].get('status_events', ())
        if (len(events) != 1 or events[0].get('phase') != 'drink_description_recovery'
                or events[0].get('source_restored') is not True or events[0].get('drinks_consumed') != 0):
            raise ValueError('HIF opening drink recovery lacks its cancel-only receipt')
        verify_unstarted_core(self._opening_resume['initial_core'], events[0].get('after_checkpoint', {}))

    def _resume_opening_status_panel(self):
        from .hif_hand_inspection import restore_hand_inspection
        from .hif_opening_status_resume import status_resume_request

        source = self._opening_source
        restore_hand_inspection(self.context, source['live_events'][-1], self.module, source=source)
        if self._opening_resume.get('status_panel_source'):
            from .hif_recovery_reports import read_recovery_report

            source = read_recovery_report(self.runtime / ('result-' + self._opening_resume['status_panel_source'] + '.json'))
        self.context.gkh_hif_unstarted_status_panel = status_resume_request(source)
        try:
            self._observe_opening_panel('status')
        finally:
            self.context.gkh_hif_unstarted_status_panel = None

    def _observe_opening_panel(self, key):
        from .opening_observer import OPENING_MODES, opening_component, opening_component_params

        start = len(self.result.get('published_components', ()))
        input_start = len(self.result['native_input_dispatches'])
        params = opening_component_params(self.params, dict(OPENING_MODES)[key])
        self.module.ProduceLiveDecisionObserve().run(self.context,
            SimpleNamespace(custom_action_param=json.dumps(params)))
        reports = self.result.get('published_components', ())[start:]
        if (len(reports) != 1 or reports[0].get('event') != 'live_status_observe'
                or reports[0].get('executed') is not False or reports[0].get('execution_submit_count') != 0
                or reports[0].get('total_input_click_count') != len(self.result['native_input_dispatches'])-input_start):
            raise ValueError('HIF continued opening observation has unaccounted input')
        self.context.gkh_hif_unstarted_components[key] = opening_component(
            reports[0], key, self.context.gkh_hif_live_context)

    def _unbound_opening_retry(self, *, maximum_actions, timeout_seconds, parent_deadline):
        """Reuse the cold loader's completed metadata and proven dispatch accounting."""
        from .hif_hand_inspection import HAND_VISITS_VERSION, TITLE_VISIT_VERSION
        from .hif_navigation_recovery import find_unstarted_menu_resume

        session = self.context.gkh_hif_input_session
        if (self.params['hif_policy_profile'] != 'hif-strong-reference-trial-v1'
                or session.record['handoff'].get('live_context') is not None):
            return None
        openings = [event for event in self.result['live_events'] if event.get('event') == 'live_opening_state']
        if len(openings) != 1 or openings[0].get('accepted') is not False:
            return None
        opening = openings[0]
        previous = self.result.get('opening_navigation_resume') or {}
        if (self.result.get('input_count') == 0 and previous.get('retained_components')
                and opening.get('components') == previous['retained_components']):
            return None
        inspections = (opening.get('extended_hand_identity') or {}).get('inspections', ())
        for inspection in (opening.get('hand_identity_inspection') or {}, *inspections):
            # fb8 already owns re-observation and any second click on its
            # failed select/read/clear. A completed earlier title can be kept.
            if inspection.get('version') in (TITLE_VISIT_VERSION, HAND_VISITS_VERSION) and inspection.get('accepted') is not True:
                return None
        try:
            recovered = find_unstarted_menu_resume(self.runtime, session, maximum_actions=maximum_actions,
                timeout_seconds=timeout_seconds, parent_deadline=parent_deadline)
        except ValueError:
            # The original STOP is authoritative for an unknown or uncertain
            # boundary. This invocation may spend only the existing window.
            return None
        if recovered is None:
            return None
        _, source, proof = recovered
        if (source['id'] != self.result['id'] or proof.get('source_result') != source['id']
                or proof.get('boundary_kind') != 'unbound_components'):
            return None
        retained = proof.get('retained_components', {})
        if any(retained.get(key) != component for key, component in opening.get('components', {}).items()):
            return None
        if not (proof.get('pending_inventory_return') or proof.get('recover_closed_inventory')
                or proof.get('resume_status_panel') or proof.get('item_tail_resume')):
            # A cold reader can also restart a partial panel. In this invocation
            # only resume if all sent inputs belong to observations we retain.
            retained_inputs = sum(event.get('total_input_click_count', 0) for event in source['live_events']
                if event.get('accepted') is True)
            metadata = source.get('opening_navigation_resume')
            if metadata is None or metadata.get('boundary_kind') == 'unstarted_playable_wait':
                retained_inputs += len(source['context_recovery']['inputs'])
            title = opening.get('hand_identity_inspection') or {}
            if proof.get('restore_opening_title') and title.get('accepted') is True:
                retained_inputs += title['input_click_count']
            if retained_inputs != source['input_count']:
                return None
        return recovered

    def _restore_unbound_opening(self, endpoints):
        from .hif_native.native_opening_root import verify_native_round_context

        source = self._opening_source
        self.context.gkh_hif_live_context = self.module.HifLiveRunContext(**source['observed_live_context'])
        self.context.gkh_hif_round_context = verify_native_round_context(source, self.module, required=True)
        self.context.gkh_hif_context_game_pid = endpoints['game']['pid']
        if self._opening_resume['boundary_kind'] in UNREAD_INVENTORY_SOURCES:
            live = deepcopy(self._opening_resume['expected_live'])
            if self._opening_resume['boundary_kind'] == 'unbound_route_entry':
                self.context.gkh_hif_unstarted_entry_live = live
            else:
                self.context.gkh_hif_unstarted_notebook_live = live
        else:
            self.context.gkh_hif_unstarted_core = deepcopy(self._opening_resume['initial_core'])
            self.context.gkh_hif_unstarted_components = deepcopy(self._opening_resume.get('retained_components', {}))
            if self._opening_resume.get('reuse_inventory_details'):
                from .inventory_customization import partial_opening_details

                self.context.gkh_hif_opening_inventory_details = partial_opening_details(source)
            if self._opening_resume.get('reread_inventory_return'):
                from .hif_closed_inventory import reobserve_partial_inventory_detail, reobserve_partial_inventory_return

                self.result['opening_inventory_return_reobservation'] = reobserve_partial_inventory_return(
                    source, self.module, self.context)
                detail = reobserve_partial_inventory_detail(source, self.module, self.context)
                if detail is not None:
                    self.result['opening_inventory_detail_reobservation'] = detail
                    saved = getattr(self.context, 'gkh_hif_opening_inventory_details', {})
                    saved[detail['zone'], detail['index'], detail['card_id']] = detail
                    self.context.gkh_hif_opening_inventory_details = saved
            self.context.gkh_hif_pending_inventory_return = self._opening_resume.get('pending_inventory_return')
            self.context.gkh_hif_item_tail_resume = self._opening_resume.get('item_tail_resume')
            self.context.gkh_hif_opening_drink_prefix = self._opening_resume.get('drink_detail_prefix')
            if self._opening_resume.get('recover_closed_inventory') or self._opening_resume.get('restore_opening_title'):
                from .hif_hand_inspection import restore_hand_inspection

                title_source = source
                if self._opening_resume.get('opening_title_source'):
                    from .hif_recovery_reports import read_recovery_report

                    title_source = read_recovery_report(self.runtime / (
                        'result-' + self._opening_resume['opening_title_source'] + '.json'))
                restore_hand_inspection(self.context, title_source['published_components'][-1], self.module, source=title_source)
            if self._opening_resume.get('recover_closed_inventory') is True:
                from .hif_closed_inventory import recover_closed_inventory

                cards = recover_closed_inventory(source, self.module, self.context,
                    self.context.gkh_hif_input_session.record['ledger']['state'])
                self.context.gkh_hif_unstarted_components = {'cards': cards}
        recovery = source['context_recovery']
        self.result['context_recovery'] = recovery
        if self._opening_resume.get('restore_drink_panel'):
            self._restore_opening_drink_panel()
        if self._opening_resume.get('resume_status_panel'):
            self._resume_opening_status_panel()
        if (self._opening_resume.get('item_tail_resume') or {}).get('open_before'):
            try:
                self._observe_opening_panel('items')
            finally:
                self.context.gkh_hif_item_tail_resume = None
        return recovery

    def _poll_battle_progress(self, image=None):
        if getattr(self.context.tasker, 'stopping', False):
            raise ValueError('HIF battle task stopped')
        progress = self._battle_progress
        if image is not None and progress.retry_network(self.context, image, click=self._battle_navigation_click):
            return True
        entering_recovery = progress.recovery_deadline is None
        progress.poll()
        if entering_recovery and progress.recovery_deadline is not None:
            self.result['battle_progress_recovery'] = '120s_without_confirmed_progress'
            self._persist()
        if image is not None:
            from .hif_upstream_ui import HIF_SKIP_RECOGNITION
            from .hif_capture_adapter import captured_at

            context = self.context
            skip = 'ProduceRecognitionSkipRound'

            def animating():
                return (getattr(context.run_recognition('ProduceRecognitionHealthFlag', image), 'hit', False)
                    and not getattr(context.run_recognition(skip, image, pipeline_override={skip: HIF_SKIP_RECOGNITION}),
                        'hit', False)
                    and not getattr(context.run_recognition('ProduceRecognitionChooseMoveCards', image), 'hit', False))

            def advance(point):
                self._battle_navigation_click((*point, 1, 1),
                    captured_at(context.tasker.controller, image), 'exam_animation')

            if progress.exam.advance_animation(context, image, is_animating=animating, click=advance):
                return True
        return False

    def _watch_unconfirmed_selection(self, first):
        """Continue the ordinary page loop after selection, without replaying the tap."""
        from .hif_card_retry import read_retry_scene, same_retry_scene, observe_after_unconfirmed
        from .hif_upstream_policy import run_strong_card_step
        from .upstream_hif_card_ui import BattleRecognitionMixin

        context = self.context
        baseline, policy = context.gkh_hif_reobserved_selection_source
        original = deepcopy(self.result['upstream_ui_journal'])
        receipt = self.result['unconfirmed_selection'] = dict(ui=original, observation=first)
        retry = original['selection_retry']
        observation = first
        rebound = False
        prior_scene_changed = False
        reconsidered = None
        last_observation = None
        remaining = baseline['core']['fields'][0]
        image = None
        try:
            while True:
                context.gkh_hif_reobserved_battle = None
                classification = observation['classification']
                receipt['current'] = observation
                core = observation['core']
                observed_state = (classification, core['fields'] if core else None,
                    core['families'] if core else None, observation.get('reason'))
                changed = observed_state != last_observation
                last_observation = observed_state
                if classification == 'terminal':
                    self._persist()
                    return True
                if getattr(context.tasker, 'stopping', False):
                    raise ValueError('HIF battle task stopped')
                if changed and classification in ('changed', 'unchanged'):
                    if core['fields'][0] != remaining:
                        remaining = core['fields'][0]
                        self._battle_progress.confirmed(remaining)
                    projection = dict(event='live_preinput_state', executed=False, execution_submit_count=0)
                    self.module._observe_hif_rule_progress(context, projection, core=core,
                        expected=policy.bundle.state, baseline=baseline, action=None, preinput=True)
                    rebound = projection.get('accepted') is True
                    receipt['preinput_observation'] = projection
                    if classification == 'changed':
                        prior_scene_changed = not rebound
                    if not rebound and classification == 'changed':
                        receipt['unresolved_state'] = dict(frame_id=observation['frame_id'],
                            reason=projection.get('reason'), missing='observed transition from the bound hand and turn to this HUD')
                    if classification == 'changed' and rebound:
                        signature = (core['fields'], core['families'])
                        if reconsidered != signature:
                            reconsidered = deepcopy(signature)
                            # Re-enter the configured policy on the observed state. If its
                            # decision is still the original card, its first tap is spent.
                            context.gkh_hif_reobserved_selected_target = dict(action=original['action'],
                                frame_id=observation['frame_id'], marker_observed=False, selection_retry=retry, ui=original)
                            returned = self.module.ProduceLiveDecisionObserve().run(context,
                                SimpleNamespace(custom_action_param=json.dumps(self.params)))
                            next_observation = getattr(context, 'gkh_hif_reobserved_battle', None)
                            if returned:
                                receipt['decision_returned'] = returned
                                return True
                            if next_observation is not None:
                                baseline, policy = context.gkh_hif_reobserved_selection_source
                                original = deepcopy(self.result['upstream_ui_journal'])
                                retry = original['selection_retry']
                                observation = next_observation
                                image = None
                                continue
                            # A failed new decision may itself have sent an action. Its
                            # standard settlement path owns any such recorded submission.
                            receipt['decision_returned'] = False
                            return False
                marker = image is not None and BattleRecognitionMixin._hif_has_select(context, image, original['target_box'])
                if marker and not rebound and not prior_scene_changed and retry.get('before'):
                    # A raised target can obscure the full hand reader. Compare it to
                    # this tap's real pre-selection HUD and visible card marks.
                    page = read_retry_scene(self.module, context, image)
                    rebound = same_retry_scene(retry['before'], page)
                    receipt['selected_scene'] = page
                if marker and rebound:
                    context.gkh_hif_reobserved_selected_target = dict(action=original['action'],
                        frame_id=observation['frame_id'], marker_observed=True, selection_retry=retry, ui=original)
                    result = run_strong_card_step(self.module, context, self.params, policy, baseline)
                    if result.get('execution_submit_count') or result.get('accepted'):
                        return result.get('accepted') is True
                    if not result.get('reobserved_battle'):
                        receipt['reason'] = result.get('reason')
                        return False
                    original = deepcopy(result['ui'])
                    observation = result['reobserved_battle']
                    image = None
                    continue
                if changed:
                    self._persist()
                time.sleep(BattleRecognitionMixin.CARD_HAND_POLL_INTERVAL)
                image = context.tasker.controller.post_screencap().wait().get()
                if self._poll_battle_progress(image) or self._close_unexpected_drink(image):
                    image = None
                    continue
                observation = observe_after_unconfirmed(self.module, context, self.params, baseline,
                    image=image, selection_retry=retry)
        finally:
            context.gkh_hif_reobserved_battle = None
            context.gkh_hif_reobserved_selected_target = None
            context.gkh_hif_reobserved_selection_source = None

    def _close_unexpected_drink(self, image):
        from .drink_panel import close_miyu_drink_panel
        from .hif_upstream_ui import upstream_drink_detail_open
        from .hif_capture_adapter import captured_at

        context = self.context
        if not upstream_drink_detail_open(context, image):
            return False
        controller = context.tasker.controller

        def fresh():
            frame = controller.post_screencap().wait().get()
            return frame, captured_at(controller, frame)

        return close_miyu_drink_panel(context, fresh=fresh, click=self._battle_navigation_click,
            frame_identifier=self.module.frame_identifier)

    def _battle_navigation_click(self, box, captured_at, phase):
        x, y, w, h = box
        _, deadline = self._navigation_timing(captured_at)
        self.result['battle_navigation'] = dict(phase=phase, captured_at=captured_at)
        return self._click(x+w//2, y+h//2, before_dispatch=lambda: self._navigation_guard(deadline),
            input_deadline=deadline, authorization_floor=captured_at, preparation_key='native_navigation_preparation').wait()

    def _new_record(self, index, *, opening_resume=None, has_business_parent=True):
        if self.automatic_control is not None:
            from .hif_live_recovery import require_budget
            require_budget(self.automatic_control)
        name = self.attempt_id if index == 1 else f'{self.attempt_id}-step-{index}'
        output = self.runtime / f'result-{name}.json'
        self.runtime.mkdir(parents=True, exist_ok=True)
        # Never overwrite a failed or uncertain transaction, including after a
        # process restart. Recovery must name the saved business source.
        with output.open('x', encoding='utf-8') as stream:
            stream.write('{}')
        if self.result is not None:
            self._completed_inputs += self.result['input_count']
        self.output = output
        self.result = dict(id=name, host_attempt=self.host_attempt, status='TRANSACTION_STARTED',
            formal_params=self.params, live_events=[], frames=[], live_raw_fields=[],
            native_input_dispatches=[], input_count=0, information_boundary_execution=True,
            handoff_id=self.context.gkh_hif_battle_handoff.handoff_id)
        from .hif_battle_process_binding import process_fields

        self.result.update(process_fields(self.context.gkh_hif_input_session.record))
        if self.params['hif_policy_profile'] == 'hif-strong-reference-trial-v1':
            from .high_score.hif_strong_reference import REFERENCE_COMMIT, REFERENCE_MODEL_SHA256, configuration

            settings = configuration()['policy_kwargs']
            self.result['policy_trial'] = dict(profile=self.params['hif_policy_profile'],
                reference_commit=REFERENCE_COMMIT, model_sha256=REFERENCE_MODEL_SHA256,
                independent_quality_passed=False, simulations=settings['simulations'],
                minimum_root_samples=settings['min_root_samples'], window_seconds=settings['window_seconds'])
        if index == 1 and self._navigation_resume is not None:
            self.result['saved_navigation_resume'] = deepcopy(self._navigation_resume)
        if index == 1:
            opening_resume = self._opening_resume
        if opening_resume is not None:
            self.result['opening_navigation_resume'] = deepcopy(opening_resume)
        if self.automatic_control is not None:
            control = self.automatic_control
            control.update(root=control['root'] or name, previous_record=self._automatic_previous_record,
                sequence=control['sequence'] + 1, windows_used=control['windows_used'] + 1)
            self._automatic_previous_record = name
        self._control_json_cache = None
        if self.records and has_business_parent and opening_resume is None:
            self.result['native_inprocess_source'] = self.records[-1]
        self.records.append(name)
        self._persist()

    def _journal(self, name, stage, payload):
        copied = json.loads(json.dumps(payload, ensure_ascii=False, default=self._encode))
        saved = self.result.setdefault(name, {})
        saved.update(copied, stage=stage)
        saved.setdefault('stages', []).append(stage)
        self._persist()

    def _component(self, event):
        # The regular logger records full decisions; this hook ensures the
        # accepted business boundary is durable before another decision starts.
        self.result.setdefault('published_components', []).append(event)
        self._persist()

    def _log(self, message):
        try:
            event = json.loads(str(message.record.get('message', '')))
        except (TypeError, ValueError):
            return
        if isinstance(event, dict) and str(event.get('event', '')).startswith('live_'):
            self.result['live_events'].append(event)

    def _cache(self, policy):
        from .hif_native.native_fast_cache import finish_secondary
        from .hif_native.native_search_cache import persist_search_cache

        if policy is None:
            # A submitted strong intent is restored only after its real
            # settlement; its complete cache was verified before observation.
            return
        profile = getattr(policy, 'profile', self.params['hif_policy_profile'])
        if profile != self.params['hif_policy_profile']:
            raise ValueError('HIF actual policy differs from its authorised mode')
        if getattr(policy, '_restored_host_calls', None) is not None:
            receipt = policy.bind_restored_host_session(self._policy_deadline())
            if receipt is not None:
                self.result['mcts_restored_host_session'] = receipt
        self.result['actual_policy_profile'] = profile
        if profile in NETWORK_PROFILES:
            finish_secondary(self.runtime, self.result, policy)
        persist_search_cache(self.runtime, self.result, policy)
        self._persist()

    def _policy_deadline(self):
        """Use the current host bound; observation time is not a search window."""
        from .hif_battle_engineering_window import window_limits

        self._require_time()
        if automatic_host_active():
            return None
        deadlines = [value for value in (self._parent_deadline,
            getattr(self, '_invocation_deadline', None)) if value is not None]
        if self.automatic_control is not None:
            control = self.automatic_control
            limit = window_limits(control)[0]
            deadlines.extend((limit, control.get('active_parent_deadline',
                control.get('parent_deadline', limit))))
        if not deadlines:
            raise ValueError('policy has no current bounded host session')
        return time.perf_counter() + min(deadlines) - time.time()

    def _strong_window_start(self, policy, request):
        from .hif_native.native_search_window import begin_window

        self.result['mcts_search_window'] = begin_window(self.runtime, self.result, policy, request)
        self._persist()

    def _verify_selected_cache(self, bundle, selected):
        from .hif_native.native_search_cache import verify_selected_cache

        report = verify_selected_cache(self.runtime, self.context.gkh_hif_mcts_selected_cache_source, bundle, selected)
        self.result['mcts_selected_cache_verified'] = report
        return report

    def _restore_selected_cache(self, policy, after, settlement):
        from .hif_native.native_handoff_policy import restore_policy
        from .hif_native.native_inprocess_resume import promote_verified_settlement

        rebound = promote_verified_settlement(policy.bundle, after, settlement)
        candidate, report = restore_policy(self.runtime, self.context.gkh_hif_mcts_selected_cache_source,
            rebound, before=policy.bundle.state, action=policy.response.action, submitted=True,
            source_drink_slots=policy.bundle.drink_slots, verified_successors=policy.committed_transition())
        self.context.gkh_hif_opening_state = rebound
        self.context.gkh_hif_mcts_policy = candidate
        self.result['mcts_search_cache_restore'] = report
        self._cache(candidate)
        return dict(report, retained=report.get('restored') is True)

    def _secondary_start(self, policy, request):
        from .hif_native.native_fast_cache import begin_secondary

        begin_secondary(self.runtime, self.result, policy, request)
        self._persist()

    def _click(self, x, y, *, before_dispatch, input_deadline, authorization_floor,
               preparation_key='native_input_preparation'):
        from .hif_live_io import post_live_click_with_deadline

        self._retain_current_frame('pre-click')
        def guarded_dispatch():
            checked_at = before_dispatch()
            # No input occurs in before_dispatch. Validate the full control
            # once after it, at the same final SDK boundary as before.
            self._require_time()
            return checked_at
        return post_live_click_with_deadline(self.controller, x, y,
            result=self.result, persist=self._persist, verify_endpoints=self._endpoints,
            before_dispatch=guarded_dispatch, input_deadline=input_deadline,
            authorization_floor=authorization_floor, not_dispatched_error=self.module._LiveInputNotDispatched,
            preparation_key=preparation_key)

    def _choice_reader(self):
        from arena_winrate.badge_reference import BadgeReferenceGallery

        from .inventory import InventoryThumbnailRecognizer
        from .inventory_customization import load_customization_catalog

        module = self.module
        recognizer = module.EmbeddingCardRecognizer.load(module.MODEL_ROOT)
        gallery = BadgeReferenceGallery.load(module._resolve_runtime_path(
            Path('assets/resource/base/model/embedding/arena_badge_reference'),
            Path('resource/base/model/embedding/arena_badge_reference')))
        return (InventoryThumbnailRecognizer(gallery, recognizer), load_customization_catalog(
            module.CATALOG_PATH, module.CATALOG_PATH.parent / 'customizations.json'))

    def _capture_choice_frame(self):
        from .hif_capture_adapter import captured_at

        controller = self.context.tasker.controller
        image = controller.post_screencap().wait().get()
        return image, captured_at(controller, image)

    def _hold(self, policy):
        from .hif_upstream_ui import wait_after_confirmed_move
        from .hif_hold_transaction import execute_hif_hold
        from .high_score.contracts import ActionKind

        action = policy.response.action
        existing = self.result.get('hold_journal')
        if existing:
            raise ValueError('HIF hold transaction already started; input cannot be replayed')
        self._journal('hold_journal', 'hold_origin_submitted', dict(
            run_id=policy.bundle.run_id, revision=policy.bundle.revision, origin=asdict(action)))
        reader, catalog = self._choice_reader()
        outcome = execute_hif_hold(self.context, policy.bundle, action, policy.committed_event_branches,
            capture=self._capture_choice_frame, frame_id=self.module.frame_identifier,
            reader=reader, catalog=catalog, journal=lambda stage, row: self._journal('hold_journal', stage, row))
        if action.kind is ActionKind.USE_DRINK:
            wait_after_confirmed_move(self.context)
        return outcome

    def _choice(self, projected, event, baseline, observe_settlement):
        from .high_score.choice import build_hif_choice_request
        from .high_score.contracts import CapabilityReport
        from .hif_choice_transaction import execute_hif_choice_transaction

        module, context = self.module, self.context
        context.gkh_hif_opening_state = projected.source_bundle
        reader, catalog = self._choice_reader()
        candidate = context.gkh_hif_mcts_policy
        chosen = []

        def choose(current, panel):
            request = build_hif_choice_request(current, card_slots=panel['card_slots'],
                target_only=panel.get('target_instance_only') is True,
                capabilities=CapabilityReport(True, True, True, True, True),
                request_id=self.result['id'] + '-retrieval')
            origin = self.result.get('choice_journal', {}).get('joint_event', {})
            submitted = (origin.get('execution_submit_count') == 1
                and sum(row.get('use_submitted') is True for row in origin.get('transactions', ())) == 1)
            decision = candidate.bind_retrieval_choice(current, request, drink_submitted=submitted)
            self.result['secondary_decision'] = decision.to_dict()
            if not decision.has_action_intent:
                raise ValueError('HIF retrieval policy stopped: ' + decision.reason)
            chosen[:] = [decision.action]
            return decision.action

        saved = self.result.get('choice_journal', {}) if self.result.get('native_choice_source') else {}
        plan = getattr(getattr(candidate, 'candidate', None), '_choice_continuation', None) or {}
        outcome = execute_hif_choice_transaction(context, projected, reader=reader,
            planned_instance=getattr(plan.get('target'), 'card_instance_id', None),
            detail_catalog=catalog,
            epoch=f'{projected.source_bundle.run_id}:{projected.source_bundle.revision}:drink:{projected.action.drink_id}',
            detail_evidence=saved.get('detail_evidence', ()),
            previous_scrolls=saved.get('choice_scrolls', ()),
            selected_choice=saved.get('choice', {}) if saved.get('stage') == 'choice_selected' else None,
            capture_frame=self._capture_choice_frame,
            journal=context.gkh_hif_choice_journal, choose_action=choose,
            observe_settlement=observe_settlement, baseline=baseline, frame_identifier=module.frame_identifier)
        event.update(outcome, event=event.get('event', 'live_choice_transaction'),
            execution_submit_count=event.get('execution_submit_count', 0))
        if outcome['accepted']:
            context.gkh_hif_choice_journal('settled', {'settlement': outcome['settlement']})
            confirmation = outcome.get('choice', {})
            context.gkh_hif_mcts_retrieval_commit = (projected, chosen[0],
                confirmation.get('confirmation_submitted') is True and confirmation.get('choice_submit_count') == 1)
        return event

    def _observe_confirmed_retrieval(self, checkpoint):
        from .hif_upstream_ui import wait_after_confirmed_move
        from .high_score.choice import build_hif_choice_request
        from .high_score.contracts import CapabilityReport

        projected = checkpoint['retrieval']
        recorded = checkpoint['row']['choice_journal']['choice']
        if callable(getattr(self.context, 'gkh_hif_poll_battle_progress', None)):
            # The confirmed move owns this page on a cold resume, too. Wait
            # for its actual return before attempting to read the new hand.
            wait_after_confirmed_move(self.context)
        settlement = self.module._observe_hif_opening_state(self.context, self.params,
            committed_expected=checkpoint['expected'], baseline_event=checkpoint['event'], committed_action=projected.action,
            confirmed_choice=recorded)
        self.result['settlement'] = settlement
        if settlement.get('accepted') is True:
            request = build_hif_choice_request(projected, card_slots=recorded['card_slots'],
                target_only=recorded.get('selected_panel', {}).get('target_instance_only') is True,
                capabilities=CapabilityReport(True, True, True, True, True), request_id=self.result['id']+'-confirmed-retrieval')
            selected = checkpoint['policy'].bind_retrieval_choice(projected, request, drink_submitted=True)
            if not selected.has_action_intent or asdict(selected.action) != recorded['action']:
                raise ValueError('confirmed retrieval differs from original joint plan')
            self.context.gkh_hif_mcts_retrieval_commit = (projected, selected.action, True)
        return dict(event='live_choice_transaction', accepted=settlement.get('accepted') is True, settlement=settlement,
            execution_submit_count=0, choice_submit_count=0, reason=settlement.get('reason'))

    def _bind_runtime(self, *, automatic_recovery=False):
        from .extended_opening import observe_extended_opening

        context = self.context
        context.gkh_hif_information_boundary_execution = True
        context.gkh_hif_gray_preview_only = False
        context.gkh_hif_mcts_retrieval_ready = True
        context.gkh_hif_mcts_cache_sink = self._cache
        strong = self.params['hif_policy_profile'] == 'hif-strong-reference-trial-v1'
        if automatic_recovery or strong:
            context.gkh_hif_upstream_card_ui = True
            # The existing pre-dispatch persistence writes this journal along
            # with its actual input audit. No extra whole-report write per frame.
            context.gkh_hif_upstream_ui_sink = lambda row: self.result.update(upstream_ui_journal=row)
            context.gkh_hif_upstream_dispatch_count = lambda: len(self.result['native_input_dispatches'])
            context.gkh_hif_upstream_preparation = lambda: self.result.get('native_upstream_input_preparation')
        if strong:
            context.gkh_hif_mcts_session_deadline = self._policy_deadline
            context.gkh_hif_mcts_window_start = self._strong_window_start
            context.gkh_hif_mcts_verify_selected_cache = self._verify_selected_cache
            context.gkh_hif_mcts_restore_selected_cache = self._restore_selected_cache
        else:
            context.gkh_hif_fast_secondary_start = self._secondary_start
        context.gkh_hif_post_live_click = self._click
        context.gkh_hif_choice_transaction = self._choice
        if automatic_recovery and self.params['hif_policy_profile'] in NETWORK_PROFILES:
            context.gkh_hif_hold_transaction = self._hold
        context.gkh_hif_component_sink = self._component
        context.gkh_hif_choice_journal = lambda stage, payload: self._journal('choice_journal', stage, payload)
        context.gkh_hif_gray_selection_journal = lambda stage, payload: self._journal('gray_selection_journal', stage, payload)
        context.gkh_hif_extended_opening_observer = lambda *args: observe_extended_opening(self.module, context, *args)
        for name in ('status', 'inventory', 'history', 'item', 'recovery'):
            setattr(context, f'gkh_hif_{name}_failure_sink',
                lambda image, label=name: self._retain_frame(image, label + '-failure'))

    @with_hif_runtime_sources
    @recovery_report_scope
    @automatic_battle_run
    def run(self, *, maximum_actions=128, timeout_seconds=7200, resume_source=None, automatic_recovery=False,
            parent_deadline=None, identity_readback=None):
        """Recover recorded timing failures without replaying a submitted action."""
        from .hif_recovery import recover_hif_live_context

        readback_receipt = None
        observe_submission_once = False

        def verify_initial_checkpoint(source, restored):
            nonlocal readback_receipt, observe_submission_once
            from .hif_live_recovery import verify_recovery_checkpoint

            verified = verify_recovery_checkpoint(source, restored,
                readback=identity_readback, runtime=self.runtime,
                context=self.context._hif_wrapped)
            observe_submission_once = verified == 'submitted_pending'
            if identity_readback is not None:
                readback_receipt = verified

        if (self._used or type(maximum_actions) is not int or not 1 <= maximum_actions <= 128
                or type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 7200
                or type(automatic_recovery) is not bool or automatic_recovery and resume_source is not None):
            raise ValueError('HIF continuous invocation cannot replay or renew its bounds')
        profile = self.context.gkh_hif_input_session.record['handoff']['policy_profile']
        if profile == UINET21_PROFILE:
            from .hif_gkms_runtime import require_uinet21_resources

            require_uinet21_resources()
        elif profile in NETWORK_PROFILES:
            from .hif_gkms_runtime import require_resources

            require_resources()
        else:
            from .high_score.hif_strong_reference import verified_model_path

            verified_model_path()
        # Only the selected policy's frozen resources; no model inference/UI.
        self._used = True
        from .hif_live_recovery import validate_parent_deadline
        validate_parent_deadline(parent_deadline)
        self._parent_deadline, self._last_parent_check = parent_deadline, time.time()
        self._require_time()
        live = self.context.gkh_hif_input_session.record['handoff'].get('live_context')
        checkpoint = None
        completed_provider = getattr(self.context, 'completed_battle_checkpoint', None)
        completed_checkpoint = (completed_provider(self.context.gkh_hif_input_session, self.module)
            if live is not None and callable(completed_provider) else None)
        automatic_failure = None
        entry_identity = None
        opening_recompute = getattr(self.context, 'gkh_hif_opening_recompute', None)
        if opening_recompute is not None:
            if not automatic_recovery or live is None or completed_checkpoint is not None:
                raise ValueError('opening recompute requires the current started battle')
            from .hif_native.native_opening_recompute import prepare_opening_recompute

            self.automatic_control, latest, checkpoint = prepare_opening_recompute(
                self.runtime, self.context.gkh_hif_input_session, self.module,
                specification=opening_recompute, attempt=self.attempt_id,
                maximum_actions=maximum_actions, timeout_seconds=timeout_seconds, parent_deadline=parent_deadline,
                authorise=None)
            self._automatic_previous_record = latest['id']
            resume_source = latest['id']
        elif automatic_recovery:
            from .hif_live_recovery import find_resume, new_control, require_new_opening
            if live is None:
                from .hif_navigation_recovery import find_completed_navigation_resume
                navigation = find_completed_navigation_resume(self.runtime,
                    self.context.gkh_hif_input_session, maximum_actions=maximum_actions,
                    timeout_seconds=timeout_seconds, parent_deadline=parent_deadline)
                if navigation is None:
                    from .hif_navigation_recovery import find_unstarted_menu_resume
                    opening = find_unstarted_menu_resume(self.runtime, self.context.gkh_hif_input_session,
                        maximum_actions=maximum_actions, timeout_seconds=timeout_seconds, parent_deadline=parent_deadline)
                    if opening is None:
                        require_new_opening(self.runtime, self.context.gkh_hif_input_session)
                        self.automatic_control = new_control(self.context.gkh_hif_input_session,
                            maximum_actions=maximum_actions, timeout_seconds=timeout_seconds, parent_deadline=parent_deadline)
                        if profile == 'hif-strong-reference-trial-v1':
                            entry_identity = getattr(self.context.gkh_hif_input_session, 'battle_entry_identity', None)
                            if entry_identity is not None:
                                entry_identity = dict(entry_identity,
                                    ownership_revision=self.context.gkh_hif_battle_handoff.ownership_revision)
                    else:
                        self.automatic_control, latest, self._opening_resume = opening
                        self._automatic_previous_record = latest['id']
                        self._opening_source = latest
                else:
                    self.automatic_control, latest, source, self._navigation_resume = navigation
                    if self._navigation_resume['boundary_kind'] != 'unbound_opening' or source != latest['id']:
                        raise ValueError('HIF unbound opening navigation changed its original STOP')
                    self._automatic_previous_record = latest['id']
            else:
                from .hif_navigation_recovery import find_completed_navigation_resume
                navigation = (None if completed_checkpoint is not None else
                    find_completed_navigation_resume(self.runtime,
                        self.context.gkh_hif_input_session, maximum_actions=maximum_actions,
                        timeout_seconds=timeout_seconds, parent_deadline=parent_deadline))
                if navigation is None:
                    self.automatic_control, latest, resume_source = find_resume(self.runtime,
                        self.context.gkh_hif_input_session, maximum_actions=maximum_actions,
                        timeout_seconds=timeout_seconds, parent_deadline=parent_deadline,
                        **({'completed_checkpoint': completed_checkpoint} if completed_checkpoint is not None else {}))
                else:
                    self.automatic_control, latest, resume_source, self._navigation_resume = navigation
                self._automatic_previous_record = latest['id']
                automatic_failure = latest if latest.get('transaction_closed') is not True else None
                if completed_checkpoint is not None:
                    proof = completed_checkpoint['completed_frontier']
                    if proof.get('continuation_source', proof['business_frontier']['pending_source']) != latest['id']:
                        raise ValueError('completed frontier is no longer the current transaction boundary')
                    checkpoint = completed_checkpoint
                    resume_source = checkpoint['row']['id']
                    automatic_failure = None
        if live is not None:
            if resume_source is None:
                raise ValueError('HIF already started; restore the saved transaction instead of a new opening')
            from .hif_native.native_handoff_resume import load_handoff_checkpoint
            if checkpoint is None:
                checkpoint = load_handoff_checkpoint(self.runtime, resume_source,
                    self.context.gkh_hif_input_session, self.module)
            if not automatic_recovery and checkpoint['row'].get('automatic_battle') is not None:
                raise ValueError('HIF automatic battle cannot bypass its cumulative budget with an explicit source')
            if automatic_failure is not None and checkpoint.get('retrieval') is None:
                if self._navigation_resume is None:
                    verify_initial_checkpoint(automatic_failure, checkpoint)
                else:
                    from .hif_navigation_recovery import verify_navigation_checkpoint
                    verify_navigation_checkpoint(self._navigation_resume, checkpoint)
        elif resume_source is not None:
            raise ValueError('HIF resume source has no durable started battle')
        if identity_readback is not None and readback_receipt is None:
            raise ValueError('HIF separate readback was not used for its explicit recovery source')
        started, sink = time.perf_counter(), None
        self._invocation_deadline = None if automatic_recovery else time.time() + timeout_seconds
        previous_environment = {}
        try:
            self._new_record(1)
            if readback_receipt is not None:
                self.result['post_commit_identity_readback'] = readback_receipt
            self.context = self.module._adapt_hif_capture_context(self.context)
            endpoints = self._endpoints(self.controller.info)
            # Preserve the legacy sidecar field, sourced from the actual endpoint.
            endpoints['game']['client_size'] = endpoints['game']['observed_client_size']
            self.endpoints = endpoints
            self.result.update(endpoints=endpoints, controller_info=self.controller.info)
            hwnd = self.context.tasker.controller.info.get('hwnd')
            if type(hwnd) is not int or hwnd <= 0:
                raise ValueError('HIF current Maa window handle is missing')
            environment = {self.module.DEBUG_TARGET_HWND_ENV: str(hwnd),
                self.module.DEBUG_TARGET_PID_ENV: str(endpoints['game']['pid'])}
            previous_environment = {key: os.environ.get(key) for key in environment}
            os.environ.update(environment)
            self._bind_runtime(automatic_recovery=automatic_recovery)
            if automatic_recovery:
                self._battle_progress = BattleProgress()
                self.context.gkh_hif_poll_battle_progress = self._poll_battle_progress
                self.context.gkh_hif_close_unexpected_drink = self._close_unexpected_drink
            sink = self.module.logger.add(self._log, format='{message}', level='WARNING')
            if checkpoint is not None:
                # The loader verified this same-game/owned-handoff identity.
                # Every resumed action still needs a fresh public-state/input
                # check; reopening unchanged notebook metadata is redundant.
                self.context.gkh_hif_live_context = self.module.HifLiveRunContext(
                    **checkpoint['row']['observed_live_context'])
                self.context.gkh_hif_context_game_pid = endpoints['game']['pid']
                recovery = checkpoint['row']['context_recovery']
            elif self._opening_resume is not None and self._opening_resume['boundary_kind'] in ('unbound_components', *UNREAD_INVENTORY_SOURCES):
                recovery = self._restore_unbound_opening(endpoints)
            else:
                expected = {}
                if self._opening_resume is not None and self._opening_resume['boundary_kind'] != 'unstarted_playable_wait':
                    expected['expected_live'] = self._opening_resume['expected_live']
                    if 'completed_phases' in self._opening_resume:
                        expected['completed_phases'] = self._opening_resume['completed_phases']
                if entry_identity is not None:
                    expected['entry_identity'] = entry_identity
                _, recovery = recover_hif_live_context(self.context,
                    round_number=self.context.gkh_hif_battle_handoff.round_number, **expected)
            self.recovery = recovery
            self.result['context_recovery'] = recovery
            if checkpoint is not None:
                from .hif_native.native_handoff_resume import bind_handoff_checkpoint
                bind_handoff_checkpoint(self.context, checkpoint, self.result, self.module)
                self._cache(checkpoint['policy'])
            opening_retry = None
            checkpoint_retry = None
            saved_opening_retried = False
            def execute_step(index):
                nonlocal checkpoint, checkpoint_retry, opening_retry, recovery, observe_submission_once
                require_hif_handoff(self.context, self.params)
                if automatic_recovery and getattr(self.context.tasker, 'stopping', False):
                    raise ValueError('HIF battle task stopped')
                if index > 1:
                    if checkpoint_retry is not None:
                        self.automatic_control, _, checkpoint = checkpoint_retry
                        checkpoint_retry = None
                    if opening_retry is None:
                        self._new_record(index, has_business_parent=checkpoint is None)
                    else:
                        self.automatic_control, self._opening_source, self._opening_resume = opening_retry
                        self._new_record(index, opening_resume=self._opening_resume)
                        self.result.update(endpoints=endpoints, controller_info=self.controller.info)
                        if hasattr(self.context, 'gkh_hif_extended_opening_identity'):
                            del self.context.gkh_hif_extended_opening_identity
                        self.context.gkh_hif_current_hand_inspection = None
                        recovery = self._restore_unbound_opening(endpoints)
                        opening_retry = None
                self.result.update(endpoints=endpoints, context_recovery=recovery)
                if index > 1 and checkpoint is not None:
                    from .hif_native.native_handoff_resume import bind_handoff_checkpoint

                    bind_handoff_checkpoint(self.context, checkpoint, self.result, self.module)
                    self._cache(checkpoint['policy'])
                if checkpoint is not None:
                    self.result['execution_live_context'] = {
                        key: value for key, value in checkpoint['live'].items() if key != 'round_number'}
                self._persist()
                previous = getattr(self.context, 'gkh_hif_committed_event', None)
                if checkpoint is not None and checkpoint.get('forced_skip'):
                    from .hif_upstream_policy import run_strong_end_turn
                    from .high_score.live_policy import BoundHifOpeningPolicy

                    candidate = self.context.gkh_hif_mcts_policy
                    bundle, baseline = checkpoint['bundle'], checkpoint['event']
                    policy = BoundHifOpeningPolicy(bundle, action_order=baseline['core']['fields'][5],
                                                  decision_policy=candidate)
                    policy.response = candidate.recover_end_turn(bundle.state)
                    self.result['hiro_forced_skip'] = dict(reason='three_no_effect_dispatches')
                    self._cache(candidate)
                    event = run_strong_end_turn(self.module, self.context, self.params, policy, baseline)
                    returned = event['accepted'] is True
                elif checkpoint is not None and checkpoint.get('retrieval') is not None:
                    projected = checkpoint['retrieval']
                    confirmed = checkpoint.get('expected') is not None
                    if confirmed:
                        observe_submission_once = True
                        event = self._observe_confirmed_retrieval(checkpoint)
                    else:
                        event = dict(event='live_choice_transaction', execution_submit_count=0)
                        try:
                            event = self._choice(projected, event, checkpoint['event'],
                                lambda expected, baseline, action: self.module._observe_hif_opening_state(self.context,
                                    self.params, committed_expected=expected, baseline_event=baseline, committed_action=action))
                        except ValueError as error:
                            from .hif_native.native_choice_resume import has_unconfirmed_choice_selection

                            if not has_unconfirmed_choice_selection(self.result.get('choice_journal', {})):
                                raise
                            event.update(accepted=False, choice_submit_count=0, reason=str(error))
                    returned = event.get('accepted') is True
                    if returned:
                        _, choice_action, choice_submitted = self.context.gkh_hif_mcts_retrieval_commit
                        checkpoint['policy'].confirm_retrieval_committed(
                            run_id=projected.source_bundle.run_id, revision=projected.source_bundle.revision,
                            before=projected.source_bundle.state, action=projected.action,
                            pending_state=projected.pending_state, choice_action=choice_action,
                            after=self.context.gkh_hif_committed_state,
                            drink_submitted=True, choice_submitted=choice_submitted, accepted=True)
                        self.context.gkh_hif_mcts_retrieval_commit = None
                        self._cache(checkpoint['policy'])
                        if confirmed:
                            self.context.gkh_hif_choice_journal('settled', {'settlement': event['settlement']})
                    self.module._publish_hif_component(self.context, event)
                elif checkpoint is not None and checkpoint.get('pending'):
                    from .hif_native.native_handoff_resume import complete_handoff_pending

                    returned = complete_handoff_pending(self.context, checkpoint, self.result,
                        self.module, self.runtime, self.params)
                else:
                    if checkpoint is not None and checkpoint.get('selected'):
                        from .hif_native.native_handoff_resume import reconcile_cleared_selection

                        reconcile_cleared_selection(self.context, checkpoint, self.result, self.module, self.params)
                        self._persist()
                    returned = self.module.ProduceLiveDecisionObserve().run(self.context,
                        SimpleNamespace(custom_action_param=json.dumps(self.params)))
                return index, previous, returned

            def observe_step(attempt):
                index, previous, returned = attempt
                reobserved = getattr(self.context, 'gkh_hif_reobserved_battle', None)
                if reobserved is not None and self._battle_progress is not None:
                    if reobserved['classification'] == 'terminal':
                        self.result['unconfirmed_selection'] = dict(observation=reobserved)
                        self.context.gkh_hif_reobserved_battle = None
                        self.context.gkh_hif_reobserved_selection_source = None
                        returned = True
                    else:
                        returned = self._watch_unconfirmed_selection(reobserved)
                self.result['live_return'] = returned
                terminal = getattr(self.context, 'gkh_hif_terminal_event', None)
                try:
                    self._retain_current_frame('observer-return')
                    self._flush_frames()
                except Exception as error:
                    if returned is not True or terminal is None:
                        raise
                    # The score receipt comes from the observed result. Failing
                    # to archive diagnostic PNGs is recorded separately.
                    self.result['frame_persistence_error'] = str(error)
                if self.automatic_control is not None:
                    self.result['observer_returned'] = dict(record=self.result['id'],
                        sequence=self.automatic_control['sequence'])
                if returned is not True:
                    outcome = ExamOutcome.REOBSERVE
                elif terminal is not None:
                    outcome = ExamOutcome.RESULT
                else:
                    outcome = ExamOutcome.PROGRESS
                return ExamCycle(outcome, (index, previous, returned, terminal))

            def recover_step(evidence):
                nonlocal checkpoint, checkpoint_retry, opening_retry, observe_submission_once
                nonlocal automatic_failure, saved_opening_retried
                index, _, _, terminal = evidence
                if self._battle_progress is not None and terminal is None:
                    self._poll_battle_progress()
                self.result['status'] = 'STOP'
                self._persist()
                if self.result.get('hold_journal'):
                    self.result['automatic_recovery_stop'] = 'hold_requires_observed_resolution_no_replay'
                    self._persist()
                    return False
                if self.result.get('pending_dispatch_outcome') == 'not_consumed':
                    from .hif_native.native_pending_outcome import prepare_unconsumed_retry, is_unconsumed_observation

                    if not is_unconsumed_observation(self.result):
                        return False
                    checkpoint = prepare_unconsumed_retry(checkpoint, self.result, self.runtime,
                        recovery=self._battle_progress.exam if self._battle_progress is not None else None)
                    if checkpoint is None:
                        self.result['automatic_recovery_stop'] = 'verified_unconsumed_retry_limit'
                        self._persist()
                        return False
                    observe_submission_once = False
                    self.result['automatic_recovery'] = dict(source=self.result['id'],
                        reason='dispatch_proven_not_consumed', retry_limit=3 if self._battle_progress else 1,
                        business_history_advanced=False)
                    self.context.gkh_hif_mcts_policy = checkpoint['policy']
                    self._cache(checkpoint['policy'])
                    self._persist()
                    return True
                if observe_submission_once:
                    return False
                if self.automatic_control is not None:
                    from .hif_live_recovery import (
                        find_resume,
                        reserve_retry,
                        recovery_candidate,
                        reported_pending_stop,
                        verify_recovery_checkpoint,
                    )
                    from .hif_native.native_choice_resume import is_confirmed_retrieval_stop
                    from .hif_native.native_observed_identity_resume import unsubmitted_drink_inspection

                    candidate = recovery_candidate(self.result)
                    drink_inspection = unsubmitted_drink_inspection(self.result) is not None
                    if candidate is not None or drink_inspection or reported_pending_stop(self.result) or is_confirmed_retrieval_stop(self.result):
                        try:
                            if candidate is not None:
                                source = reserve_retry(self.automatic_control, self.result, previous=automatic_failure)
                                automatic_failure = self.result
                                self.result['automatic_recovery'] = dict(source=source,
                                    reason=candidate['kind'], business_input_replayed=False)
                                self._persist()
                            else:
                                control, latest, source = find_resume(self.runtime,
                                    self.context.gkh_hif_input_session, maximum_actions=maximum_actions,
                                    timeout_seconds=timeout_seconds, parent_deadline=parent_deadline)
                                if latest['id'] != self.result['id'] or source != latest['id']:
                                    raise ValueError('HIF settlement retry differs from the reported submission')
                            from .hif_native.native_handoff_resume import load_handoff_checkpoint

                            restored = load_handoff_checkpoint(self.runtime, source,
                                self.context.gkh_hif_input_session, self.module)
                            if candidate is None and restored.get('receipt', {}).get('boundary_kind') == 'confirmed_retrieval':
                                observe_submission_once = True
                                checkpoint_retry = control, latest, restored
                            elif candidate is None and drink_inspection:
                                verify_recovery_checkpoint(latest, restored)
                                checkpoint_retry = control, latest, restored
                            elif candidate is not None:
                                verify_recovery_checkpoint(self.result, restored)
                                checkpoint = restored
                            else:
                                verified = verify_recovery_checkpoint(latest, restored)
                                if restored.get('pending') is not True:
                                    raise ValueError('HIF settlement retry lacks a submitted action')
                                observe_submission_once = verified == 'submitted_pending'
                                checkpoint_retry = control, latest, restored
                        except Exception as error:
                            self.result['automatic_recovery_stop'] = str(error)
                            return False
                        return True
                    if not saved_opening_retried:
                        from .hif_native.native_opening_intent_resume import prepare_saved_opening, is_original_saved_opening_intent

                        if is_original_saved_opening_intent(self.result):
                            try:
                                checkpoint_retry = prepare_saved_opening(self.runtime,
                                    self.context.gkh_hif_input_session, self.module, self.result,
                                    attempt=f'{self.attempt_id}-step-{index+1}', maximum_actions=maximum_actions,
                                    timeout_seconds=timeout_seconds, parent_deadline=parent_deadline,
                                    authorise=None)
                            except (KeyError, ValueError):
                                return False
                            saved_opening_retried = True
                            return True
                    opening_retry = self._unbound_opening_retry(maximum_actions=maximum_actions,
                        timeout_seconds=timeout_seconds, parent_deadline=parent_deadline)
                    if opening_retry is not None:
                        return True
                return False

            def commit_step(evidence):
                nonlocal observe_submission_once, automatic_failure, checkpoint
                _, previous, returned, terminal = evidence
                observe_submission_once = False
                if terminal is not None:
                    session = self.context.gkh_hif_input_session
                    current_screen = getattr(session, 'current_screen', False) is True
                    self.result.update(terminal_settlement=terminal, status='TERMINAL_VERIFIED', transaction_closed=True,
                        continuation_records=list(self.records), elapsed_ms=(time.perf_counter() - started) * 1000)
                    if current_screen:
                        self.result.update(current_battle_finished=True, returned_to_nonbattle=False)
                    else:
                        self.result.update(handoff_return_prepared=True,
                            handoff_ownership_revision=session.record['ownership']['revision'])
                    self._persist()
                    self._sealed_terminal_source = self.output
                    return_hif_battle_handoff(self.context, live_return=returned,
                        terminal_source=self.result['id'], terminal_runtime=self.runtime)
                    if not current_screen:
                        self.result['returned_to_nonbattle'] = True
                    return self.result
                settlement = getattr(self.context, 'gkh_hif_committed_event', None)
                policy = getattr(self.context, 'gkh_hif_mcts_policy', None)
                from .hif_native.native_handoff_policy import phase

                if (not isinstance(settlement, dict) or settlement is previous or settlement.get('accepted') is not True
                        or getattr(self.context, 'gkh_hif_committed_state', None) is None
                        or policy is None or phase(policy) != 'observed_successor' or policy._pending is not None):
                    raise ValueError('HIF continuation lacks a newly settled, committed policy state')
                automatic_failure = None
                if self._battle_progress is not None:
                    confirmed = settlement.get('rule_progress', {}).get('action', {}).get('kind')
                    components = self.result.get('published_components', ())
                    played_or_used = confirmed in ('PLAY_CARD', 'USE_DRINK') or any(
                        row.get('accepted') is True and (row.get('drinks_consumed') or
                            row.get('executed') and row.get('action', {}).get('kind') == 'PLAY_CARD')
                        for row in components)
                    self._battle_progress.confirmed(
                        getattr(self.context.gkh_hif_committed_state, 'turns_remaining', None),
                        played_or_used=played_or_used)
                    self._poll_battle_progress()
                self.result.update(status='RULE_MAINTAINED_NATIVE_TRANSACTION', transaction_closed=True)
                self.context.gkh_hif_gray_selected_source = None
                checkpoint = None
                self._persist()
                return None

            def invocation_expired():
                if time.perf_counter() - started >= timeout_seconds:
                    self.result['continuation_stop'] = 'invocation_timeout'
                    return True
                return False

            run_exam_session(execute=execute_step, observe=observe_step, commit=commit_step, recover=recover_step,
                limit=None if automatic_recovery else maximum_actions,
                expired=None if automatic_recovery else invocation_expired,
                on_limit=lambda: self.result.update(continuation_stop='invocation_action_limit'))
        except Exception as error:
            if self.result is None:
                raise
            self.result.update(status='STOP', error=str(error), error_type=type(error).__name__)
            if getattr(error, 'event', None) is not None:
                self.result['failure_event'] = error.event
        finally:
            for key, value in previous_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if sink is not None:
                self.module.logger.remove(sink)
            if self.result is not None:
                try:
                    self._flush_frames()
                except Exception as error:
                    self.result['frame_persistence_error'] = str(error)
                self.result.update(continuation_records=self.records, elapsed_ms=(time.perf_counter() - started) * 1000)
                self._persist()
        return self.result
