"""Observe the current interval and its configured customization entry.

Audited hiro v2.3.3 sets every ProduceHIFCardCustomOpt_* to false and
_handle_main exits when that queue is empty. No purchase is inferred here.
"""
from __future__ import annotations

import time

import cv2

from .host import HifRoutePage
from .engine import Target, Outcome, Decision, Observation, propose_unique, validate_observation
from .ledger import LedgerEffect

# Existing zero-cost navigation receipts; no card selection or paid execution.
NAVIGATION_DESTINATIONS = {
    'leave_interval': ('interval_end_confirm', 'battle_entry'),
    'confirm_interval_exit': ('battle_intro', 'battle_entry'),
    'battle_intro_next': ('battle_entry',),
    'customization_open_list': ('card_customize_list',),
}


def read_interval_confirmation(rows, *, image=None, after_exit=False):
    from .production import _row, _text
    from .nonbattle_observation import _within
    # hiro's normal shop.tap_end recognizes the white cancel button after
    # opening the exit dialog. Its action always targets the right side.
    cancel_visible = False
    if after_exit and getattr(image, 'shape', ()) == (1280, 720, 3):
        colors = cv2.cvtColor(image[1138:1182, 110:260], cv2.COLOR_BGR2HSV)
        _, saturation, brightness, _ = cv2.mean(colors)
        cancel_visible = saturation <= 30 and brightness >= 215
    if cancel_visible:
        control = next((row for row in rows if row['text'] in ('本戦へ', '決定', 'はい', 'OK')
            and _within(row['box'], (360, 1080, 360, 160))), None)
        if control is None:
            target = (500, 1158, 1, 1)
        else:
            x, y, width, height = control['box']
            target = (x + width // 2, y + height // 2, 1, 1)
    else:
        _text(rows, r'インターバル終了確認', (25, 740, 650, 85))
        target = tuple(_row(rows, r'本戦へ', (450, 1120, 150, 90))['box'])
    # The modal can obscure the HUD. Keep independent reads when available.
    try:
        current, maximum = map(int, _text(rows, r'\d{1,2}/\d{1,2}', (285, 35, 165, 85)).split('/'))
    except ValueError:
        current = maximum = None
    try:
        currency = int(_text(rows, r'\d{1,6}', (350, 80, 110, 75)))
    except ValueError:
        currency = None
    return dict(scene='interval_end_confirm', stamina=current, max_stamina=maximum,
        currency=currency, target=target)


def navigation_effect(state, key):
    suffix = 'leave-interval' if key == 'leave_interval' else key
    return LedgerEffect(f'{state.run_id}:{state.revision}:{suffix}', key)


def read_hif_battle_introduction(rows, *, context=None, image=None):
    from .production import _row, _text
    _text(rows, r'現在順位', (10, 140, 220, 90))
    next_page = _row(rows, r'タップして次へ', (240, 1140, 260, 95))
    return dict(scene='battle_intro', target=tuple(next_page['box']), stamina=None, anchor=next_page)


class HifSupplyProducer:
    def __init__(self, recovery):
        self.parent = recovery
        self.context, self.host, self.clock = recovery.context, recovery.host, recovery.clock
        self.last_frame_ids = ()

    def _page(self, image):
        from live_decision.hif_recovery import _field, read_recovery_page

        from .production import _current_rows, read_interval_page, read_bound_drink_slots
        rows = _current_rows(self.context, image)
        from .customization_pixels import read_customization_list, read_interval_customization_entry
        try:
            customization_list = read_customization_list(rows)
        except ValueError:
            customization_list = None
        if customization_list is not None:
            state = self.host.ledger.state
            if ((customization_list['stamina'], customization_list['max_stamina'], customization_list['currency'])
                    != (state.stamina, state.max_stamina, state.currency)):
                raise ValueError('customization list resources changed from the complete current ledger')
            return customization_list
        try:
            introduction = read_hif_battle_introduction(rows, context=self.context, image=image)
        except ValueError as error:
            if any(row['text'] == '現在順位' and row['score'] >= .95 for row in rows):
                raise ValueError('current HIF ranking introduction remains unresolved: ' + str(error)) from error
            introduction = None
        if introduction is not None:
            introduction['maintained_resource_revision'] = self.host.ledger.state.revision
            return introduction
        ledger = self.host.ledger
        after_exit = (ledger.pending.effect.target_key == 'leave_interval' if ledger.pending
            else bool(ledger.settled and ledger.settled[-1].rsplit(':', 1)[-1] == 'leave-interval'))
        try:
            confirmation = read_interval_confirmation(rows, image=image, after_exit=after_exit)
        except ValueError:
            confirmation = None
        if confirmation is not None:
            state = self.host.ledger.state
            if any(confirmation[key] is not None and confirmation[key] != getattr(state, key)
                    for key in ('stamina', 'max_stamina', 'currency')):
                raise ValueError('interval confirmation resources differ from their complete baseline')
            return confirmation
        try:
            root = read_interval_page(self.context, image, rows=rows, menu=False)
        except ValueError:
            root = None
        if root is not None:
            state = self.host.ledger.state
            if ((root['stamina'], root['max_stamina'], root['currency'])
                    != (state.stamina, state.max_stamina, state.currency)):
                raise ValueError('interval resources changed from the complete current ledger')
            root['slots'] = read_bound_drink_slots(self.context, image, state.drinks,
                footer_offset=13, maintained_revision=state.revision)
            end = _field(self.context, image, 'IntervalEnd', (624, 1045, 94, 75), r'終了')
            try:
                entry = read_interval_customization_entry(image, rows)
            except ValueError:
                entry = None
            return dict(scene='interval', target=tuple(end['box']),
                stamina=root['stamina'], max_stamina=root['max_stamina'], currency=root['currency'], drink_slots=root['slots'],
                customization_remaining=entry['remaining'] if entry else None,
                customization_target=entry['target'] if entry else None)
        state = self.host.ledger.state
        terminal = (self.host.handoff or {}).get('terminal_receipt')
        if terminal is not None and (not isinstance(terminal, dict)
                or type(terminal.get('round_number')) is not int or terminal['round_number'] != 1):
            raise ValueError('battle entry has no supported next-round provenance')
        round_number = 1 if terminal is None else 2
        # Handoff consumes only the Live HUD, never a menu input target.
        page = read_recovery_page(self.context, image, round_number=round_number, require_live_menu=False)
        if page['phase'] != 'live':
            raise ValueError('interval exit did not reach a supported battle entry')
        if type(page['stamina']) is not int or not 0 <= page['stamina'] <= state.max_stamina:
            raise ValueError('battle entry Live stamina exceeds its verified current maximum')
        # Only R1 has the documented base recovery of floor(maximum/2).
        # An uncapped HIF recovery bonus needs its own observed source; it
        # cannot be invented from an arbitrary value inside 0..maximum.
        recovery = state.max_stamina // 2 if round_number == 1 else 0
        expected = min(state.max_stamina, state.stamina + recovery)
        if page['stamina'] != expected:
            raise ValueError('battle entry stamina lacks its exact round-specific recovery source')
        return dict(scene='battle_entry', turns=page['turns'], stamina=page['stamina'], target=None,
            stamina_source='independently_observed_live_field',
            maintained_nonbattle_revision=state.revision,
            entry_recovery=dict(round_number=round_number, historical_stamina=state.stamina,
                base_recovery=recovery, expected=expected, observed=page['stamina']))

    def observe(self, *, reconciliation=False):
        from .hif import active_hif_role
        host = self.host
        identity = (host.handoff or {}).get('recovery_identity', {})
        if (host.ledger is None or host.ownership['owner'] != 'task600'
                or host.ownership['phase'] != 'nonbattle'
                or identity.get('run_id') != host.run_id or (identity.get('idol'), identity.get('plan')) != active_hif_role(host) or identity.get('max_stamina') != host.ledger.state.max_stamina):
            raise ValueError('current interval lacks the bound identity and complete ledger')
        deadline = self.clock.now() + 8000
        if not reconciliation and host.hooks.transition_gate.pending is not None:
            gate = host.hooks.transition_gate
            deadline = gate.issued_at_ms + gate.timeout_ms
        while True:
            started = self.clock.now()
            if started >= deadline or self.context.tasker.stopping:
                raise ValueError('current interval observation deadline reached')
            job = self.context.tasker.controller.post_screencap().wait()
            if not job.succeeded:
                raise ValueError('current interval capture failed')
            image = job.get()
            key = f'route-frame-{self.parent.game_pid}-{started}'
            try:
                page = self._page(image)
            except ValueError as error:
                from .produce_settings import ProduceSettingsObserved

                if isinstance(error, ProduceSettingsObserved):
                    raise
                metadata = dict(frame_id=key, phase='unknown', reason=str(error))
                if self.parent.evidence_sink:
                    self.parent.evidence_sink(image, metadata)
                self.parent.events.append(metadata)
                if host.hooks.transition_gate.pending is not None and self.clock.now() < deadline:
                    time.sleep(.1)
                    continue
                raise
            if self.parent.evidence_sink:
                self.parent.evidence_sink(image, dict(frame_id=key, phase=page['scene'], observation=page))
            break
        keys = (key,)
        self.last_frame_ids = keys
        revision = self.clock.now()
        key = ('leave_interval' if page['scene'] == 'interval' else
            'battle_intro_next' if page['scene'] == 'battle_intro' else 'confirm_interval_exit')
        target = Target(key, key, available=True)
        targets = () if page['scene'] in ('battle_entry', 'card_customize_list') else (target,)
        boxes = {target.key: page['target']} if targets else {}
        if page['scene'] == 'interval' and (page.get('customization_remaining') or 0) > 0:
            boxes['customization_open_list'] = tuple(page['customization_target'])
            targets += (Target('customization_open_list', 'customization_open_list', available=True),)
        page_stamina = host.ledger.state.stamina if page['scene'] == 'interval_end_confirm' else page['stamina']
        observation = Observation(host.run_id, revision, page['scene'],
            f'interval:{page["scene"]}:{page.get("turns", "")}:{page_stamina}:', 7,
            *active_hif_role(host), revision - started, len(keys), 'HIF', targets)
        effects = tuple(navigation_effect(host.ledger.state, item.key) for item in targets)
        receipt = None
        if host.ledger.pending is not None:
            pending = host.ledger.pending
            # Only this exact, zero-resource navigation can maintain inventory.
            prior = pending.effect.target_key
            if (prior not in NAVIGATION_DESTINATIONS
                    or pending.effect != navigation_effect(host.ledger.state, prior)):
                raise ValueError('interval navigation cannot settle another resource transaction')
            if prior == 'customization_open_list':
                if page['scene'] == 'card_customize_list':
                    receipt = pending.expected
            elif (page['scene'] == 'battle_entry' or prior == 'confirm_interval_exit' and page['scene'] == 'battle_intro'
                    or prior == 'leave_interval' and page['scene'] == 'interval_end_confirm'):
                receipt = pending.expected
        self.parent.events.append(dict(phase=page['scene'], frame_ids=list(keys),
            capture_age_ms=observation.capture_age_ms, customization_configuration='resolved_during_decision'))
        return HifRoutePage(observation, boxes,
            baseline=host.ledger.state, frame_ids=keys, effects=effects,
            receipt=receipt, receipt_frame_ids=keys if receipt is not None else ())

    def decide(self, observation, config):
        error = validate_observation(observation)
        if error:
            return Decision(Outcome.STOP, error)
        if observation.scene == 'battle_entry':
            return Decision(Outcome.PAUSE, 'battle_requires_task275')
        if observation.scene == 'card_customize_list':
            return Decision(Outcome.STOP, 'customization_current_card_grid_pixels_not_calibrated')
        if observation.scene == 'battle_intro':
            return propose_unique(observation, observation.targets, 'hif-current-ranking-introduction', ('battle_entry',))
        if observation.scene == 'interval_end_confirm':
            enter = tuple(target for target in observation.targets if target.kind == 'confirm_interval_exit')
            return propose_unique(observation, enter, 'hif-confirm-interval-exit', ('battle_intro', 'battle_entry'))
        if observation.scene != 'interval':
            return Decision(Outcome.STOP, 'unsupported_interval_scene')
        from .customization_runtime import queue_for_ledger, selection_for_run
        selection = selection_for_run(self.host, config, self.host._page,
            catalog=self.host._customization_catalog)
        queue = queue_for_ledger(selection, self.host.ledger)
        self.parent.events.append(dict(phase='customization_queue', enabled=selection.enabled,
            profession=selection.profession, source=selection.source,
            queue=[dict(card_id=row.card_id, instance_id=row.instance_id, status=row.status) for row in queue]))
        if any(row.status == 'ambiguous_instances' for row in queue):
            return Decision(Outcome.STOP, 'customization_physical_instances_ambiguous')
        if any(row.status == 'pending' for row in queue):
            entry = tuple(target for target in observation.targets if target.kind == 'customization_open_list')
            if entry:
                return propose_unique(observation, entry,
                    'hif-configured-customization-current-entry', ('card_customize_list',))
            # A configured target never supplies unobserved menu coordinates.
            return Decision(Outcome.STOP, 'customization_current_selection_page_required')
        return propose_unique(observation, tuple(target for target in observation.targets if target.kind == 'leave_interval'),
            'hiro-v2.3.3-empty-customization-queue-exit', NAVIGATION_DESTINATIONS['leave_interval'])

    def customization_menu(self, page, config):
        from .customization_runtime import selection_for_run, menu_for_selection
        selection = selection_for_run(self.host, config, page, catalog=self.host._customization_catalog)
        return menu_for_selection(selection, self.host.ledger, page.customization)

    def prepare(self, gate, report):
        proof = self.observe(reconciliation=True)
        return self.host.reconcile_interval_confirmation(gate, report, proof)
