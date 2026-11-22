"""Handle a Produce settings modal under the original nonbattle input lease."""
from __future__ import annotations

import time
from copy import deepcopy

from .ownership import write_hif_record, locked_hif_record
from .maa_bridge import MAX_INPUT_AGE_MS, RouteInputNotDispatched
from .produce_settings import next_setting, read_produce_settings

KEY = 'produce_settings_navigation'
ENTRY = 'ProduceHifAdvanceClick'


def pending_overlay(record, key):
    return ((record.get('handoff') or {}).get('nonbattle_context') or {}).get(key)


def pending_settings(record):
    return pending_overlay(record, KEY)


class NonbattleOverlay:
    key = KEY

    def __init__(self, runtime, *, sleep=time.sleep):
        self.runtime, self.context, self.sleep = runtime, runtime.bound, sleep
        self.record = None
        self.intent = None

    def _screen(self):
        from .production import _current_rows

        if not self.runtime.authorise_capture() or self.context.tasker.stopping:
            raise ValueError('produce settings capture is no longer authorised')
        captured_at = self.runtime.clock.now()
        job = self.context.tasker.controller.post_screencap().wait()
        if job.succeeded is not True:
            raise ValueError('produce settings capture failed')
        image = job.get()
        rows = _current_rows(self.context, image, date_probe=False, settings_probe=False)
        sink = getattr(self.runtime, 'evidence_sink', None)
        if sink is not None:
            sink(image, dict(frame_id=f'route-frame-{self.runtime.game_pid}-{captured_at}', phase=self.key, rows=rows))
        return captured_at, image, rows

    def _save(self, state):
        candidate = deepcopy(self.record)
        if candidate.get('handoff') is None:
            candidate['handoff'] = {}
        candidate['handoff'].setdefault('nonbattle_context', {})[self.key] = deepcopy(state)
        write_hif_record(self.runtime.path, candidate)
        self.record = candidate

    def validate_dispatch(self, args, kwargs):
        # Called by the product Context after coordinate mapping and its
        # current administrator endpoint verification.
        intent = self.intent
        saved = (pending_overlay(self.record, self.key) or {}).get('input')
        if (intent is None or intent != saved or intent['result'] == 'sent'
                or self.runtime._read() != self.record or self.context.tasker.stopping
                or not 0 <= self.runtime.clock.now() - intent['capture_ms'] <= MAX_INPUT_AGE_MS):
            raise ValueError('nonbattle overlay input lost its current prepared source')
        mapped = self.context.gkh_hif_capture_adapter._rect(intent['box'])
        if args != (ENTRY,) or kwargs != {'pipeline_override': {ENTRY: {'action': 'Click', 'target': mapped}}}:
            raise ValueError('nonbattle overlay input differs from its prepared control')
        return True

    def _click(self, state, control, box, captured_at):
        if not self.runtime.authorise_input() or self.context.tasker.stopping:
            raise ValueError('nonbattle overlay input is no longer authorised')
        self.intent = dict(control=control, box=list(box), capture_ms=captured_at, result='prepared')
        state['input'] = self.intent.copy()
        self._save(state)
        try:
            result = self.context.run_action(ENTRY, pipeline_override={ENTRY: {'action': 'Click', 'target': list(box)}})
        except RouteInputNotDispatched:
            state['input']['result'] = 'unissued'
            self._save(state)
            raise
        finally:
            self.intent = None
        state['input']['result'] = 'sent' if getattr(result, 'success', None) is True else 'uncertain'
        self._save(state)
        if state['input']['result'] != 'sent':
            raise ValueError('nonbattle overlay input result is unknown')


class SettingsNavigation(NonbattleOverlay):
    def _resume(self, state, dialog):
        prior = state.get('input')
        if prior is None:
            return True
        control = prior['control']
        same = dialog is not None and state['dialog'] == dialog['dialog']
        if prior['result'] != 'unissued':
            switch = dialog['switches'].get(control) if same else None
            observed = not same or switch is not None and switch['value'] is True
            if prior['result'] != 'sent' and not observed:
                raise ValueError('produce settings input remains unresolved')
            if control == 'confirm' and same:
                if self.runtime.clock.now() - prior['capture_ms'] < 5000:
                    return False
                raise ValueError('produce settings confirm has no observed dismissal')
            state['completed'].append(control)
        state['history'].append(dict(prior, dialog=state['dialog'], same_dialog_after=same))
        state['input'] = None
        self._save(state)
        return True

    def run(self):
        """Return to the original business observer once the modal goes away."""
        with locked_hif_record(self.runtime.path):
            self.record = self.runtime._read()
            if self.record['ownership']['owner'] != 'task600':
                raise ValueError('produce settings require the original nonbattle owner')
            parent = self.context.gkh_hif_route_host
            self.context.gkh_hif_route_host = self
            try:
                state = deepcopy(pending_settings(self.record)) or dict(dialog=None, completed=[], history=[], input=None)
                offset = len(state['history'])
                while True:
                    captured_at, _, rows = self._screen()
                    dialog = read_produce_settings(rows)
                    if not self._resume(state, dialog):
                        self.sleep(.2)
                        continue
                    if dialog is None:
                        state.update(dialog=None, completed=[])
                        self._save(state)
                        return state['history'][offset:]
                    if state['dialog'] != dialog['dialog']:
                        state.update(dialog=dialog['dialog'], completed=[])
                    action = next_setting(dialog, state['completed'])
                    if action is None:
                        raise ValueError('produce settings confirm is not visible')
                    self._click(state, *action, captured_at)
                    self.sleep(1 if action[0] == 'confirm' else .4)
            finally:
                self.context.gkh_hif_route_host = parent
