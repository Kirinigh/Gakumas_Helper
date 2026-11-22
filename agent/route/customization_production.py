"""Capture named results and reuse the proven interval/inventory readers.

Unknown customization menus supply no click target. Their selected option,
price and physical-card controls still need current pixel calibration.
"""
from __future__ import annotations

from dataclasses import asdict

from . import customization_transaction as transaction
from .hif import active_hif_role
from .host import HifRoutePage
from .engine import Target, Observation


class HifCustomizationProducer:
    def __init__(self, parent):
        self.parent = parent
        self.inventory = None
        self.inventory_scope = None

    def _read(self, image, rows, value):
        from .nonbattle_observation import parse_customization_success
        host = self.parent.host
        card = host._catalog.entries[value['card_id']]
        if parse_customization_success(rows, card_name=card.name):
            return dict(kind='result', scene='card_customize_result')
        if value['stage'] != 'census':
            raise ValueError('customization requires its original named success result')
        from .production import read_nonbattle_inventory
        if any(row['score'] >= .95 and row['text'] == '所持スキルカード' for row in rows):
            page = read_nonbattle_inventory(rows)
            if page['phase'] != 'inventory':
                raise ValueError('customization census requires its skill-card inventory')
            from .nonbattle_inventory import bind_inventory_census
            saved = bind_inventory_census(self, ['customization', value['transaction_id']], value.get('inventory'))
            if self.inventory is None:
                from .nonbattle_inventory import FlatInventoryCollector, load_flat_inventory_readers
                self.inventory = FlatInventoryCollector(*load_flat_inventory_readers())
                self.inventory.census_session = self.inventory_scope
                if saved is not None:
                    try:
                        self.parent._restore_inventory(collector=self.inventory,
                            retained=saved, phase='customization_inventory')
                    except BaseException:
                        self.inventory = None
                        raise
            return dict(kind='inventory', scene='inventory', target=page['target'],
                target_kind='recovery_close', target_key='recovery_close')
        from .production import read_interval_page, read_bound_drink_slots
        try:
            interval = read_interval_page(self.parent.context, image, rows=rows, menu=False)
        except ValueError as error:
            raise ValueError('customization_current_menu_pixels_not_calibrated') from error
        expected = host.ledger.pending.expected
        if (interval['currency'], interval['stamina'], interval['max_stamina']) != (
                expected.currency, expected.stamina, expected.max_stamina):
            raise ValueError('customization interval resources differ from its original submission')
        slots = read_bound_drink_slots(self.parent.context, image, expected.drinks,
            footer_offset=13, maintained_revision=host.ledger.pending.before.revision)
        result = dict(kind='interval', scene='interval', currency=interval['currency'],
            stamina=interval['stamina'], max_stamina=interval['max_stamina'], drinks=expected.drinks,
            drink_slots=slots)
        if (value.get('inventory') or {}).get('complete') is not True:
            from .production import recognize_route_inventory_button as recognize_inventory_button
            matches, _, _, _ = recognize_inventory_button(self.parent.context, image,
                tuple(self.parent.context.gkh_hif_client_size))
            if len(matches) != 1:
                raise ValueError('customization current inventory control is not uniquely observed')
            result.update(target=tuple(matches[0].box), target_kind='recovery_inventory', target_key='recovery_inventory')
        return result

    def observe(self, *, reconciliation=False):
        from .production import _current_rows
        from .customization import observe_customization_result
        parent, host = self.parent, self.parent.host
        value = transaction.request(host)
        if value is None:
            raise ValueError('customization observation has no original pending transaction')
        gate = host.hooks.transition_gate
        deadline = parent.clock.now() + 8000
        if gate.pending is not None and not reconciliation:
            deadline = min(deadline, gate.issued_at_ms + gate.timeout_ms)
        started = parent.clock.now()
        if started >= deadline or parent.context.tasker.stopping:
            raise ValueError('customization observation reached its original deadline')
        capture = parent.context.tasker.controller.post_screencap().wait()
        if capture.succeeded is not True:
            raise ValueError('customization current screenshot failed')
        image = capture.get()
        key = f'route-frame-{parent.game_pid}-{started}'
        rows = _current_rows(parent.context, image, date_probe=False)
        try:
            page = self._read(image, rows, value)
        except ValueError as error:
            metadata = dict(frame_id=key, phase='customization_unknown', reason=str(error), rows=rows)
            parent.events.append(metadata)
            if parent.evidence_sink: parent.evidence_sink(image, metadata)
            raise
        if parent.evidence_sink:
            retained = dict(page)
            if 'drinks' in retained:
                retained['drinks'] = [asdict(drink) for drink in retained['drinks']]
            parent.evidence_sink(image, dict(frame_id=key, phase='customization_' + page['kind'], observation=retained, rows=rows))
        keys = (key,)
        result = None
        if page['kind'] == 'inventory':
            index = (int(gate.pending.target.key.split(':')[1]) if gate.pending is not None
                and gate.pending.target.kind == 'recovery_inventory_detail' else None)
            census = self.inventory.observe(parent.context, [image], [rows], keys, inspect_index=index)
            if index is not None and census.get('selected_index') != index:
                raise ValueError('customization physical detail is not yet independently confirmed')
            page['inventory'] = dict(census, census_id=value['transaction_id'])
            navigation = self.inventory.next_navigation()
            if navigation is not None:
                page.update(target=navigation['target'], target_kind=navigation['kind'], target_key=navigation['key'])
        elif page['kind'] == 'result':
            card = host.ledger.pending.effect.remove_cards[0]
            result = observe_customization_result(card, frame_ids=keys, rows_by_frame=[rows],
                card_catalog=host._catalog)
        boxes = {page['target_key']: tuple(page['target'])} if page.get('target') is not None else {}
        revision = parent.clock.now()
        observation = Observation(host.run_id, revision, page['scene'],
            f'customization:{value["transaction_id"]}:{page["kind"]}:{page.get("inventory", {}).get("displacement")}',
            value['day'], *active_hif_role(host), revision-started, 1, 'HIF',
            tuple(Target(key, page['target_kind'], available=True) for key in boxes))
        parent.last_frame_ids = keys
        parent.events.append(dict(phase='customization_' + page['kind'], frame_ids=list(keys),
            capture_age_ms=observation.capture_age_ms))
        return HifRoutePage(observation, boxes, frame_ids=keys, customization_result=result,
            customization_receipt=dict(page, frame_ids=list(keys)))

    def prepare(self, gate, report):
        proof = self.observe(reconciliation=True)
        return self.parent.host.reconcile_customization_navigation(gate, report, proof)
