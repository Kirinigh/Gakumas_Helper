"""Read-only inventory navigation inside a frozen date-interrupted exchange.

The recovery producer is only a read model. Its shadow recovery phase and
observed inventory never replace the actual owner or either original gate.
"""
from __future__ import annotations

import re
import json
from copy import deepcopy
from types import SimpleNamespace
from collections import Counter
from dataclasses import replace

import cv2

from .hif import active_hif_role, HifConfig
from .host import HifRouteHost
from .engine import Target, Outcome, Decision, TransitionGate, propose_unique, source_frames_match
from .checkpoint import restore_transition, transition_checkpoint
from .maa_bridge import MAX_NAVIGATION_INPUT_AGE_MS, HifRuntimeHooks, run_hif_sequence
from .input_receipt import persist_input_event, original_input_report

EXPECTED = {'recovery_inventory': ('inventory',), 'recovery_inventory_detail': ('inventory',),
    'recovery_inventory_scroll': ('inventory',), 'recovery_close': ('resume_home',)}


def recognize_exchange_inventory_button(context, image, client_size):
    """Keep the existing complete bag template on the dialog's grey footer."""
    from .production import recognize_route_inventory_button
    if getattr(image,'shape',()) != (1280,720,3):
        raise ValueError('exchange inventory footer is outside the calibrated image')
    pixels = image.astype('int16')
    mask = ((pixels.min(2) > 145) & (pixels.max(2)-pixels.min(2) < 25)).astype('uint8')*255
    projected = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    return recognize_route_inventory_button(context,projected,client_size)


def exchange_pending(ledger, gate):
    pending = ledger.pending
    return (pending is not None and gate.pending is not None
        and gate.pending.target.kind == gate.pending.target.key == pending.effect.target_key == 'execute_lesson_exchange'
        and gate.pending.source.scene == pending.before.phase == 'card_exchange_selected'
        and gate.pending.source.day == pending.before.day in (1, 4)
        and len(pending.effect.remove_cards) == len(pending.effect.add_cards) == 1
        and not pending.before.drinks and not pending.before.counters
        and not pending.effect.remove_drinks and not pending.effect.add_drinks
        and not pending.effect.counter_deltas
        and (pending.effect.currency_delta, pending.effect.stamina_delta, pending.effect.max_stamina_delta) == (0,0,0))


def classify_census(pending, inventory, catalog, resources):
    """Compare every actual variant and duplicate, without inventing instances."""
    if (not isinstance(inventory, dict) or inventory.get('complete') is not True
            or type(inventory.get('total')) is not int
            or inventory['total'] != len(inventory.get('slots', ()))
            or [row.get('index') for row in inventory['slots']] != list(range(inventory['total']))):
        raise ValueError('exchange recovery needs the complete actual permanent census')
    actual = []
    for row in inventory['slots']:
        entry = catalog.entries.get(row.get('card_id'))
        customs = row.get('customizations')
        if (entry is None or row.get('presence') not in ('CONFIDENT_ZERO', 'DETAIL_CANDIDATE')
                or not isinstance(customs, dict)
                or any(not isinstance(key, str) or type(count) is not int or count < 1 for key, count in customs.items())
                or row['presence'] == 'CONFIDENT_ZERO' and customs
                or row['presence'] == 'DETAIL_CANDIDATE' and (row.get('detail') or {}).get('nonbattle_schema') != 1):
            raise ValueError('exchange recovery census has an unresolved permanent variant')
        actual.append((row['card_id'], entry.upgraded, tuple(sorted(customs.items()))))
    observed = Counter(actual)
    matches = []
    for name, state in (('before', pending.before), ('after', pending.expected)):
        expected = Counter((card.card_id, card.upgraded, card.customization) for card in state.cards)
        if observed == expected and resources == (state.stamina, state.max_stamina, state.currency):
            matches.append(name)
    if len(matches) != 1:
        raise ValueError('exchange recovery actual census is unknown or cannot distinguish before from after')
    return matches[0]


def reconcile_before_exchange(host, gate, report, page, events):
    """Resolve a proved absent effect; retain its entire original transaction."""
    from .ledger import HifLedger
    from .nonbattle_date import KEY, business_scope, navigation_closed
    record = json.loads(host.path.read_bytes())
    date = record['handoff']['nonbattle_context'].get(KEY) or {}
    census = (date.get('business_receipt') or {}).get('exchange_census') or {}
    saved = date.get('exchange_census') or {}
    source = {frame.get('frame_id') for frame in report.get('frames', ())}
    if (not exchange_pending(host.ledger, gate) or len(events) != 1 or not navigation_closed(record)
            or date.get('business_scope') != business_scope(record)
            or saved.get('status') != 'complete' or saved.get('proof') != census
            or census.get('classification') != 'before' or census.get('schema_version') != 1
            or census.get('transaction_id') != host.ledger.pending.effect.transaction_id
            or census.get('run_id') != host.run_id or census.get('game_pid') != host.game_pid
            or census.get('resource_frame_ids') != date['business_receipt']['frame_ids']
            or len(census.get('resource_frame_ids', ())) != 1
            or not date['business_receipt']['observation']['page_id'].startswith(
                f'date-exchange-census:{host.ledger.pending.effect.transaction_id}:')
            or page.observation.run_id != host.run_id or page.observation.day != host.ledger.state.day
            or page.observation.scene != 'card_choice'
            or page.observation.page_id.startswith('date-exchange-census:')
            or (page.observation.idol, page.observation.plan, page.observation.scenario) != (*active_hif_role(host), 'HIF')
            or not source_frames_match(page.observation, page.frame_ids)
            or page.observation.revision <= gate.pending.source.revision
            or source.intersection(page.frame_ids) or source.intersection(census.get('resource_frame_ids', ()))):
        raise ValueError('unselected exchange cannot discard its original transaction without a complete current census')
    if classify_census(host.ledger.pending, census.get('inventory'), host.ledger.card_catalog,
            tuple(census.get('resources', ()))) != 'before':
        raise ValueError('unselected exchange census does not prove the original effect absent')
    before = host.ledger.pending.before
    state = replace(before, revision=page.observation.revision, phase='card_choice')
    result = HifLedger(state, card_catalog=host.ledger.card_catalog, drink_ids=host.ledger.drink_ids,
        settled=host.ledger.settled)
    receipt = dict(outcome='verified_exchange_effect_absent_after_date_census', source_result=report['id'],
        original_transition=transition_checkpoint(gate, host.clock.epoch),
        original_ledger_pending=deepcopy(record['ledger']['pending']), exchange_census=deepcopy(census),
        receipt_frame_ids=list(page.frame_ids))
    old_ledger, old_handoff = host.ledger, host.handoff
    host.ledger = result
    host.handoff = dict(host.handoff or {}, recovery_navigation_receipts=[
        *(host.handoff or {}).get('recovery_navigation_receipts', ()), receipt])
    next_gate = TransitionGate(timeout_ms=gate.timeout_ms)
    try:
        host._save(transition_checkpoint(next_gate,host.clock.epoch))
    except BaseException:
        host.ledger, host.handoff = old_ledger, old_handoff
        raise
    return next_gate


class DateExchangeCensus:
    validate_dispatch = HifRouteHost.validate_dispatch
    _matches_prepared_business_record = HifRouteHost._matches_prepared_business_record
    _prepared_business_record = HifRouteHost._prepared_business_record

    def __init__(self, date):
        from .production import HifRecoveryProducer
        self.date = date
        self.context, self.path, self.clock = date.context, date.path, date.clock
        self.run_id, self.game_pid = date.run_id, date.game_pid
        self.ledger, self.ownership = date.ledger, date.ownership
        original = restore_transition(date.state['business_scope']['transition'], clock_epoch=self.clock.epoch)
        if not exchange_pending(self.ledger, original):
            raise ValueError('date census cannot replace another original business transaction')
        self.transaction_id = self.ledger.pending.effect.transaction_id
        saved = date.state.get('exchange_census')
        if saved is not None and (saved.get('schema_version') != 1
                or saved.get('transaction_id') != self.transaction_id or saved.get('status') not in ('active', 'complete')):
            raise ValueError('date exchange census belongs to another transaction')
        self.state = deepcopy(saved) if saved is not None else dict(schema_version=1, status='active',
            transaction_id=self.transaction_id, started_at_ms=self.clock.now(),
            transition=transition_checkpoint(TransitionGate(), self.clock.epoch), observer={}, inputs=[], confirmations=[])
        self.handoff = deepcopy(self.state['observer'])
        self._active, self._page, self.input_events = False, None, []
        self._navigation_only = False
        self._input_adapter = date._input_adapter
        self.hooks = HifRuntimeHooks(self.observe, self._allowed, lambda: self._active and self._allowed(),
            transition_gate=restore_transition(self.state['transition'], clock_epoch=self.clock.epoch),
            save_transition=self._save, cancel_unissued=self._cancel, clock=self.clock.now, clock_epoch=self.clock.epoch,
            decide=self.decide, input_event=self._event, input_result=self._result,
            max_input_age_ms=MAX_NAVIGATION_INPUT_AGE_MS)
        # This private adapter lets the established inventory collector run.
        # The original nonbattle owner and pending ledger stay on self/date.
        self.reader = SimpleNamespace(run_id=self.run_id, game_pid=self.game_pid, path=self.path, clock=self.clock,
            ownership=dict(self.ownership, phase='recovery'), ledger=self.ledger, handoff=self.handoff,
            hooks=self.hooks, _catalog=self.ledger.card_catalog)
        self.observer = HifRecoveryProducer(self.context, run_id=self.run_id, game_pid=self.game_pid, clock=self.clock,
            evidence_sink=date.runtime.evidence_sink, capture_guard=self._allowed, resume_probe_home=True)
        self.observer.host = self.reader
        self._original_page = self.observer._page
        self.observer._page = self._read_page
        self._input_source = self.reader._input_source = self.observer.freeze_input_source

    def _allowed(self, capture_endpoints=None):
        record = self.date._read()
        self.date._guard(record)
        if record['handoff']['nonbattle_context'].get('route_date_navigation') != self.date.state:
            raise ValueError('exchange census date journal changed')
        return True

    def _input_age_limit(self, proposal):
        return MAX_NAVIGATION_INPUT_AGE_MS

    def _write(self, state, *, receipt_only=False):
        if not receipt_only: self._allowed()
        self.date._write(dict(self.date.state, exchange_census=deepcopy(state)), receipt_only=receipt_only)
        self.state = deepcopy(state)

    def _save(self, checkpoint):
        self._write(dict(self.state, transition=deepcopy(checkpoint), observer=deepcopy(self.handoff)))
        return True

    def _cancel(self):
        return self._save(transition_checkpoint(TransitionGate(timeout_ms=self.hooks.transition_gate.timeout_ms), self.clock.epoch))

    def _receipt(self, envelope):
        carrier = deepcopy(envelope['handoff']['nonbattle_context']['route_input_receipt'])
        journal = list(self.state['inputs'])
        if journal and journal[-1]['original_transition'] == carrier['original_transition']:
            journal[-1] = carrier
        else:
            journal.append(carrier)
        self._write(dict(self.state, transition=deepcopy(envelope['transition']),
            observer=deepcopy(envelope['handoff']), inputs=journal), receipt_only=True)

    def _event(self, event):
        self.input_events.append(event)
        persist_input_event(self, event, write_record=self._receipt)
        self.reader.handoff = self.handoff

    def _result(self, event):
        persist_input_event(self, event, complete=True, write_record=self._receipt)
        self.reader.handoff = self.handoff

    def _dispatch_record(self):
        self._allowed()
        return json.loads(json.dumps(dict(schema_version=3, run_id=self.run_id, ownership=self.ownership,
            ledger=self.ledger.record(), transition=self.state['transition'], handoff=self.state['observer']), allow_nan=False))

    def _read_page(self, image):
        from .daily import complete_daily_resources, read_lesson_exchange_offer
        from .production import _current_rows, read_nonbattle_inventory
        rows = _current_rows(self.context, image, date_probe=False)
        self.observer.rows = rows
        try:
            read_nonbattle_inventory(rows)
        except ValueError:
            pass
        else:
            return self._original_page(image, rows=rows)
        rows, evidence = complete_daily_resources(self.context, image, rows)
        page = read_lesson_exchange_offer(rows, day=self.ledger.state.day)
        if not page['has_unselected_hint']:
            raise ValueError('exchange census lacks its actual restored unselected dialog')
        self.home = page
        return dict(phase='resume_home', target=None, day=page['day'],
            **{key: page[key] for key in ('stamina', 'max_stamina', 'currency')}, recovery_evidence=dict(resources=evidence))

    def observe(self, *, reconciliation=False):
        from .production import maintain_empty_nonbattle_drinks
        self._allowed()
        self.reader.handoff = self.handoff
        page = self.observer.observe(reconciliation=reconciliation)
        self.handoff = self.reader.handoff
        if not source_frames_match(page.observation, page.frame_ids):
            raise ValueError('exchange census page lacks its current capture')
        if page.observation.scene == 'resume_home':
            image = self.observer._source_frames[page.frame_ids[0]][0]
            inventory = self.observer.current_inventory()
            if (inventory or {}).get('complete') is not True:
                matches, _, _, _ = recognize_exchange_inventory_button(self.context, image, tuple(self.context.gkh_hif_client_size))
                if len(matches) != 1:
                    raise ValueError('exchange census inventory control is not uniquely observed')
                kind = 'recovery_inventory'
                page = replace(page, observation=replace(page.observation, targets=(Target(kind,kind,available=True),)),
                    boxes={kind: tuple(matches[0].box)})
            else:
                keys = inventory.get('frame_ids', ())
                captures = [re.fullmatch(r'route-frame-'+str(self.game_pid)+r'-([0-9]+)', key) for key in keys]
                if (not keys or len(set(keys)) != len(keys) or not all(captures)
                        or any(int(match[1]) < self.state['started_at_ms'] for match in captures)):
                    raise ValueError('exchange census contains an old or foreign inventory frame')
                resources = tuple(self.home[key] for key in ('stamina', 'max_stamina', 'currency'))
                classification = classify_census(self.ledger.pending, inventory, self.ledger.card_catalog, resources)
                slots = maintain_empty_nonbattle_drinks(self.context, image, (),
                    footer_offset=0, revision=self.ledger.state.revision)
                if len(slots) != 4 or any(slot['state'] != 'EMPTY' for slot in slots):
                    raise ValueError('exchange census has an unexpected drink or unresolved slot')
                self.proof = dict(schema_version=1, transaction_id=self.transaction_id, run_id=self.run_id,
                    game_pid=self.game_pid, classification=classification, inventory=deepcopy(inventory),
                    resources=list(resources), drink_slots=slots, resource_frame_ids=list(page.frame_ids))
        page = replace(page, observation=replace(page.observation,
            page_id=f'date-exchange-census:{self.transaction_id}:{page.observation.page_id}'))
        self._page = page
        return page.observation, page.boxes

    def decide(self, observation, config):
        if observation.scene == 'resume_home' and not observation.targets:
            self._write(dict(self.state, status='complete', proof=deepcopy(self.proof), observer=deepcopy(self.handoff)))
            return Decision(Outcome.PAUSE, 'date_exchange_actual_census_complete')
        if len(observation.targets) != 1 or observation.targets[0].kind not in EXPECTED:
            return Decision(Outcome.STOP, 'date_exchange_census_unknown_navigation')
        target = observation.targets[0]
        if target.kind == 'recovery_close' and (self.observer.current_inventory() or {}).get('complete') is not True:
            return Decision(Outcome.STOP, 'date_exchange_census_incomplete_before_close')
        return propose_unique(observation, observation.targets, 'date-exchange-census:'+target.kind, EXPECTED[target.kind])

    def _prepare(self):
        gate = self.hooks.transition_gate
        if gate.pending is None: return
        if gate.pending.target.kind not in EXPECTED or gate.pending.expected_scenes != EXPECTED[gate.pending.target.kind]:
            raise ValueError('date exchange census has an unknown pending navigation')
        report = original_input_report(self, gate)
        self.observer.recovery_report = report
        observation, _ = self.observe(reconciliation=True)
        if (not gate.pending.source.page_id.startswith(f'date-exchange-census:{self.transaction_id}:')
                or observation.scene not in gate.pending.expected_scenes
                or observation.revision <= gate.pending.source.revision
                or set(self._page.frame_ids).intersection(row.get('frame_id') for row in report['frames'])):
            raise ValueError('date exchange census lacks its fresh original navigation destination')
        self.state = dict(self.state, confirmations=[*self.state['confirmations'], dict(
            original_transition=transition_checkpoint(gate,self.clock.epoch), source_result=report['id'],
            frame_ids=list(self._page.frame_ids))])
        self._save(transition_checkpoint(TransitionGate(timeout_ms=gate.timeout_ms),self.clock.epoch))
        gate.pending, gate.issued_at_ms, gate.stopped = None, None, False

    def run(self):
        self._active = True
        previous = self.context.gkh_hif_route_host
        try:
            self.context.gkh_hif_route_host = self
            self._save(self.state['transition'])
            self._prepare()
            result = run_hif_sequence(self.context, HifConfig(mode='advance'), self.hooks, max_actions=None, timeout_ms=None)
            if result.reason != 'date_exchange_actual_census_complete': raise ValueError(result.reason)
            return self._page, self.proof
        finally:
            self.context.gkh_hif_route_host = previous
            self._active = False
