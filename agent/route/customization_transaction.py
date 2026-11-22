"""Keep one customization reserved through its independent inventory census."""
from __future__ import annotations

import re
from copy import deepcopy

from .hif import active_hif_role
from .engine import Outcome, Decision, propose_unique, validate_observation
from .customization import CustomizationMenu
from .customization_receipt import (
    CustomizationAfterMenu,
    _frames,
    _pending,
    begin_customization_receipt,
    confirm_customization_success,
    resolve_customization_receipt,
)

# These contracts grant no coordinates. The current pixel producer must supply
# a unique observed target; an absent/unrecognised menu cannot be navigated.
NAVIGATION = {
    'customization_result_ack': ('card_customize_result', ('card_customize',)),
    'customization_return_interval': ('card_customize', ('interval',)),
    'recovery_inventory': ('interval', ('inventory',)),
    'recovery_inventory_detail': ('inventory', ('inventory',)),
    'recovery_inventory_scroll': ('inventory', ('inventory',)),
    'recovery_close': ('inventory', ('interval',)),
    'customization_open_list': ('interval', ('card_customize_list',)),
    'customization_select_card': ('card_customize_list', ('card_customize',)),
}


def request(host, *, ledger=None):
    value = (host.handoff or {}).get('nonbattle_context', {}).get('customization_receipt')
    if value is None or isinstance(value, dict) and value.get('stage') in ('complete', 'cancelled'):
        return None
    owner = host.ownership or {}
    if (not isinstance(value, dict) or value.get('stage') not in ('awaiting_result', 'census')
            or value.get('run_id') != host.run_id or value.get('game_pid') != host.game_pid
            or owner.get('owner') != 'task600' or owner.get('phase') != 'nonbattle'
            or owner.get('game_pid') != host.game_pid or value.get('ownership_revision') != owner.get('revision')):
        raise ValueError('customization receipt has a different current owner or run')
    _pending(value, host.ledger if ledger is None else ledger)
    return value


def with_request(handoff, value):
    candidate = deepcopy(handoff or {})
    candidate.setdefault('nonbattle_context', {})['customization_receipt'] = deepcopy(value)
    return candidate


def _navigation(value, pending):
    if pending is None:
        return True
    kind, key = pending.target.kind, pending.target.key
    contract = NAVIGATION.get(kind)
    return (contract is not None and pending.source.scene == contract[0]
        and pending.expected_scenes == contract[1] and pending.source.day == value['day']
        and (re.fullmatch(r'inventory-detail:\d+', key) is not None if kind == 'recovery_inventory_detail' else
            re.fullmatch(r'inventory-scroll:\d+', key) is not None if kind == 'recovery_inventory_scroll' else key == kind))


def permits_navigation(host, ledger, gate):
    value = request(host, ledger=ledger)
    return value is not None and value['stage'] == 'census' and _navigation(value, gate.pending)


def start_request(host, ledger, gate):
    from .checkpoint import transition_checkpoint
    page = host._page
    if (request(host) is not None or page is None or gate.pending is None
            or gate.pending.target.kind != 'execute_customization'
            or page.observation != gate.pending.source or page.baseline != ledger.pending.before):
        raise ValueError('customization submission has no exact original menu')
    value = begin_customization_receipt(ledger, page.observation, page.customization,
        catalog=host._customization_catalog, game_pid=host.game_pid, issued_at_ms=gate.issued_at_ms)
    value.update(ownership_revision=host.ownership['revision'],
        original_transition=transition_checkpoint(gate, host.clock.epoch))
    return value


def _page(host, page, value):
    seen, proof = page.observation, page.customization_receipt
    ticks = _frames(page.frame_ids, game_pid=host.game_pid, after=value['issued_at_ms'], before=seen.revision)
    if (page is not host._page or not isinstance(proof, dict) or proof.get('frame_ids') != list(page.frame_ids)
            or seen.run_id != host.run_id or seen.day != value['day']
            or (seen.idol, seen.plan, seen.scenario) != (*active_hif_role(host), 'HIF')
            or validate_observation(seen) is not None or not 0 <= host.clock.now() - ticks[0] <= 1000
            or not 0 <= seen.capture_age_ms + host.clock.now() - seen.revision <= 1000):
        raise ValueError('customization census page lacks its independent fresh current frames')
    kinds = {'card_customize_result': 'result', 'card_customize': 'menu', 'interval': 'interval',
        'inventory': 'inventory', 'card_customize_list': 'list'}
    if kinds.get(seen.scene) != proof.get('kind'):
        raise ValueError('customization census page kind contradicts its current scene')
    if proof['kind'] in ('menu', 'interval'):
        expected = host.ledger.pending.expected
        if (any(type(proof.get(key)) is not int for key in ('currency', 'stamina', 'max_stamina'))
                or (proof['currency'], proof['stamina'], proof['max_stamina'], proof.get('drinks'))
                    != (expected.currency, expected.stamina, expected.max_stamina, expected.drinks)):
            raise ValueError('customization census resources contradict the original single submission')
    if proof['kind'] == 'menu' and (not isinstance(page.customization, CustomizationMenu)
            or (page.customization.card_instance_id, page.customization.card_id) != (value['card_instance_id'], value['card_id'])):
        raise ValueError('customization census menu belongs to another physical card')
    return proof


def _inventory(host, page, value):
    proof = page.customization_receipt
    census = proof.get('inventory')
    if (not isinstance(census, dict) or census.get('census_id') != value['transaction_id']
            or not isinstance(census.get('frame_ids'), list)
            or not set(page.frame_ids).issubset(census['frame_ids'])):
        raise ValueError('customization census differs from its original transaction')
    from .nonbattle_inventory import retain_inventory_registration
    return retain_inventory_registration(host, page, census)


def navigation_checkpoint(host, gate):
    value = request(host)
    if value is None or value['stage'] != 'census' or not _navigation(value, gate.pending) or gate.pending is None:
        raise ValueError('customization cannot navigate another unresolved transaction')
    proof = _page(host, host._page, value)
    if host._page.observation != gate.pending.source:
        raise ValueError('customization navigation source changed')
    if gate.pending.target.kind == 'customization_select_card' and proof.get('card_instance_id') != value['card_instance_id']:
        raise ValueError('customization list does not identify its original physical card')
    result = deepcopy(value)
    if proof.get('kind') == 'inventory':
        result['inventory'] = _inventory(host, host._page, value)
    result['navigation_source_frame_ids'] = list(host._page.frame_ids)
    return result


def confirm_page(host, page):
    from .input_receipt import original_input_report
    value, route = request(host), host.hooks.transition_gate.pending
    if value is None or route is None:
        raise ValueError('customization result has no original submitted action')
    proof = _page(host, page, value)
    original = original_input_report(host, host.hooks.transition_gate)
    _frames(page.frame_ids, game_pid=host.game_pid, after=host.hooks.transition_gate.issued_at_ms,
        before=page.observation.revision)
    if set(page.frame_ids).intersection((*value['source_frame_ids'], *value.get('navigation_source_frame_ids', ()))):
        raise ValueError('customization destination reused its original input source frames')
    if route.target.kind == 'execute_customization':
        if value['stage'] != 'awaiting_result' or proof.get('kind') != 'result':
            raise ValueError('customization needs its original named success result')
        result = confirm_customization_success(value, host.ledger, page.customization_result)
        if page.customization_result.frame_ids != page.frame_ids:
            raise ValueError('customization success frames differ from the current page')
        result['execution_receipt'] = original
        return result
    if value['stage'] != 'census' or not _navigation(value, route) or page.observation.scene not in route.expected_scenes:
        raise ValueError('customization census reached an unrelated navigation destination')
    result = deepcopy(value)
    if page.observation.scene == 'inventory':
        result['inventory'] = _inventory(host, page, value)
        census = result['inventory']
        if route.target.kind == 'recovery_inventory_detail' and census.get('selected_index') != int(route.target.key.split(':')[1]):
            raise ValueError('customization census opened a different physical detail')
        if route.target.kind == 'recovery_inventory_scroll':
            from .nonbattle_inventory import flat_scroll_measurement_matches
            registration = census.get('registration') or {}
            if (not flat_scroll_measurement_matches(value.get('inventory') or {}, census)
                    or registration.get('receipt_frame_ids') != list(page.frame_ids)
                    or not set(registration.get('source_frame_ids', ())).intersection(value.get('navigation_source_frame_ids', ()))):
                raise ValueError('customization census scroll lacks its original current registration')
    elif page.observation.scene == 'card_customize':
        menu = page.customization
        if not isinstance(menu, CustomizationMenu) or (menu.card_instance_id, menu.card_id) != (value['card_instance_id'], value['card_id']):
            raise ValueError('customization return menu belongs to another physical card')
    return result


def decide(host, observation, config):
    value, page = request(host), host._page
    if value is None or value['stage'] != 'census' or page is None or page.observation != observation:
        return Decision(Outcome.STOP, 'customization_original_census_reservation_required')
    proof = _page(host, page, value)
    if proof.get('kind') == 'menu' and (value.get('inventory') or {}).get('complete') is True:
        host.bind_observed_customization_receipt(page)
        return Decision(Outcome.PAUSE, 'hif_observed_customization_census_complete')
    permitted = [target for target in observation.targets if target.kind in NAVIGATION
        and NAVIGATION[target.kind][0] == observation.scene]
    # Before census completion an interval can only open the inventory. After
    # completion it can only reopen the original card through its observed list.
    if observation.scene == 'interval':
        kind = 'customization_open_list' if (value.get('inventory') or {}).get('complete') is True else 'recovery_inventory'
        permitted = [target for target in permitted if target.kind == kind]
    elif observation.scene == 'inventory':
        complete = (proof.get('inventory') or {}).get('complete') is True
        permitted = [target for target in permitted if (target.kind == 'recovery_close') == complete]
    elif observation.scene == 'card_customize_list' and proof.get('card_instance_id') != value['card_instance_id']:
        return Decision(Outcome.STOP, 'customization_current_original_card_selection_required')
    destinations = {NAVIGATION[target.kind][1] for target in permitted}
    if len(destinations) != 1:
        return Decision(Outcome.STOP, 'customization_current_census_navigation_required')
    return propose_unique(observation, tuple(permitted), 'hif-customization-original-census', destinations.pop())


def resolve(host, page):
    value = request(host)
    proof = _page(host, page, value)
    if proof.get('kind') != 'menu' or not isinstance(page.customization, CustomizationMenu):
        raise ValueError('customization settlement needs its observed post-census menu')
    after = CustomizationAfterMenu(host.run_id, page.observation.day, host.game_pid, page.observation.revision,
        page.frame_ids, page.customization, proof.get('currency'), proof.get('stamina'), proof.get('max_stamina'),
        proof.get('drinks'))
    return resolve_customization_receipt(value, host.ledger, value.get('inventory'), after,
        catalog=host._customization_catalog, now_ms=host.clock.now())
