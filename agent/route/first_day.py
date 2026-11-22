"""Observe the initial HIF deck and resources on the bound first day."""
from copy import deepcopy

import cv2

from .hif import known_hif_role
from .start_transaction import _budget, read_start_created

FIRST_DAY_CONTRACTS = {'recovery_inventory': ('start_created', ('inventory',)),
    'recovery_inventory_scroll': ('inventory', ('inventory',)),
    'recovery_close': ('inventory', ('start_created',))}


def first_day_scope(host):
    state = _budget(host)
    role = state.get('request', {}).get('identity') or ()
    if (state['status'] != 'complete' or len(role) != 2 or not known_hif_role(*role)
            or state.get('creation_receipt', {}).get('countdown_days') != 6
            or not isinstance(state.get('ap_receipt'), dict)):
        raise ValueError('initial ledger requires this same confirmed first day')
    return state


def uses_initial_deck_observation(host):
    from .new_run import start_record
    if host is None or host.ledger is not None:
        return False
    start = start_record(host.handoff) or {}
    return (start.get('status') == 'complete' and start.get('run_id') == host.run_id
        and start.get('creation_task') is not None)


def initial_deck_scroll_needs_read(host, gate):
    if (gate is None or gate.pending is None or gate.pending.target.kind != 'recovery_inventory_scroll'
            or not uses_initial_deck_observation(host)):
        return False
    inventory = (host.handoff or {}).get('nonbattle_inventory') or {}
    return not inventory.get('read_finished') and inventory.get('scroll_issued_at') != gate.issued_at_ms


def deck_view_changed(first, second):
    # hiro 8380622a deck_review._still compares only the scrolling card area.
    before = cv2.cvtColor(first[395:1010, :720], cv2.COLOR_BGR2GRAY)
    after = cv2.cvtColor(second[395:1010, :720], cv2.COLOR_BGR2GRAY)
    return cv2.mean(cv2.absdiff(before, after))[0] >= 2.0


class InitialDeckScrolling(ValueError):
    pass


def observe_initial_deck(parent, images, rows, keys, *, settle_deadline=None, still_moving=False):
    from .recovery_inventory import retained, current_record
    from .nonbattle_inventory import FlatInventoryCollector, flat_inventory_count, load_flat_inventory_readers
    previous = retained(parent.host)
    gate = parent.host.hooks.transition_gate
    scroll = gate.pending is not None and gate.pending.target.kind == 'recovery_inventory_scroll'
    if previous and previous.get('first_day_once'):
        if previous['read_finished'] or not scroll or previous.get('scroll_issued_at') == gate.issued_at_ms:
            return previous
    try:
        if still_moving:
            raise ValueError('flat inventory scroll has not settled')
        if parent.inventory is None:
            parent.inventory = FlatInventoryCollector(*load_flat_inventory_readers())
            if previous:
                parent._restore_inventory(retained=previous)
        value = parent.inventory.observe(parent.context, images, rows, keys,
            allow_unknown_badges=True, allow_unknown_total=True)
    except (OSError, RuntimeError, ValueError) as error:
        if (settle_deadline is not None and parent.clock.now() < settle_deadline and
                str(error) in ('flat inventory scroll has not settled', 'inventory scroll has no unique overlapping translation')):
            raise InitialDeckScrolling(str(error)) from error
        # A read can fail before any identity is available. The displayed count
        # is still independent evidence from the current viewport.
        try:
            counts = {flat_inventory_count(group) for group in rows}
        except ValueError:
            counts = set()
        value = deepcopy(previous) if previous else dict(total=counts.pop() if len(counts) == 1 else None,
            displacement=0, visible_boxes={}, slots=[], selected_index=None, complete=False, frame_ids=list(keys))
        value.update(read_error=str(error), unreadable_frame_ids=list(keys))
    ids = [slot['card_id'] for slot in value['slots']]
    all_identified = value['total'] == len(ids) and all(card_id is not None for card_id in ids)
    # hiro's six screens include the first.
    screens = 1
    if scroll and previous and previous.get('first_day_once'):
        screens += previous['review_screens']
    finished = (screens >= 6 or not value['visible_boxes'] or bool(value.get('read_error')) or value['total'] == len(ids)
        or scroll and previous is not None and value['displacement'] <= previous['displacement'])
    value.update(first_day_once=True, census_session=deepcopy(parent.inventory_scope),
        review_screens=screens, read_finished=finished,
        spotlight_plus=True if '762' in ids else False if all_identified and finished else None)
    if scroll:
        value['scroll_issued_at'] = gate.issued_at_ms
    result = current_record(parent.host, value)
    parent.host.handoff = dict(parent.host.handoff or {}, nonbattle_inventory=result)
    if finished:
        start = first_day_scope(parent.host)
        start['initial_deck'] = dict(spotlight_plus=value['spotlight_plus'], total=value['total'],
            identified=sum(card_id is not None for card_id in ids),
            frame_ids=list(dict.fromkeys([*value['frame_ids'], *value.get('unreadable_frame_ids', ())])))
        parent.host.handoff['cultivation_start'] = start
    return result


def read_first_day_page(parent, image, rows):
    from .calendar import read_route_resources
    from .production import _text, read_interval_empty_drinks, recognize_route_inventory_button
    start = first_day_scope(parent.host)
    retained = parent.prepare_inventory_census()
    observed_once = (retained or {}).get('first_day_once') is True and retained.get('read_finished') is True
    if not observed_once and parent.inventory is None and isinstance(retained, dict) and retained.get('complete') is True:
        from .nonbattle_inventory import FlatInventoryCollector, load_flat_inventory_readers
        inventory = FlatInventoryCollector(*load_flat_inventory_readers())
        inventory.census_session = parent.inventory_scope
        inventory.total, inventory.displacement = retained['total'], retained['displacement']
        inventory.slots = {row['index']: deepcopy(row) for row in retained['slots']}
        inventory.selected_index = retained['selected_index']
        if not inventory.complete:
            raise ValueError('initial retained inventory is not complete')
        parent.inventory = inventory
    if observed_once or parent.inventory is not None and parent.inventory.complete:
        page = read_start_created(rows)
        return dict(page, **page['creation_resources'], counters={},
            drink_slots=read_interval_empty_drinks(parent.context, image, footer_offset=0))

    # This visit only opens the deck. The creation receipt already holds its
    # resources; the returned calendar supplies a fresh HUD for the ledger.
    _text(rows, r'H[.]I[.]F本戦まで', (20, 30, 190, 65))
    _text(rows, r'6日', (40, 75, 145, 90))
    prior = start.get('initial_deck') or {}
    unopened = prior.get('opened') is False
    matches = () if unopened else recognize_route_inventory_button(parent.context, image,
        tuple(parent.context.gkh_hif_client_size))[0]
    pending = parent.host.hooks.transition_gate.pending
    if (not matches and uses_initial_deck_observation(parent.host) and retained is None and (not prior or unopened)
            and (pending is None or pending.target.kind == 'start_resume_current_save')):
        from .daily import read_day_one_lesson
        current = read_day_one_lesson(image, rows)
        if current['scene'] == 'lesson':
            resources = {field: current[field] for field in ('stamina', 'max_stamina', 'currency')}
            return dict(phase='start_created', countdown_days=6, target=None, inventory_opened=False,
                creation_resources=resources, **resources, counters={},
                drink_slots=read_interval_empty_drinks(parent.context, image, footer_offset=0))
    if len(matches) != 1:
        raise ValueError('current first-day owned-card control is not uniquely observed')
    page = dict(phase='start_created', countdown_days=6, target=tuple(matches[0].box), inventory_navigation=True)
    try:
        page['start_evidence'] = read_route_resources(rows)
    except ValueError:
        pass  # An unreadable HUD does not change the current deck button.
    return page
