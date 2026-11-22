"""Resume the unread fourth item after a verified, closed opening inspection."""
from copy import deepcopy


def is_drink_stop_after_item_return(row):
    events = row.get('live_events') or ()
    tail = (row.get('opening_navigation_resume') or {}).get('item_tail_resume') or {}
    return (bool(tail.get('open_before')) and len(events) == 3 and events[0].get('accepted') is True
        and events[-1].get('reason') == 'HIF opening component failed: drinks')


def resume_drinks_after_item_return(record, previous, proof, row, *, counted_return):
    from types import SimpleNamespace

    from .opening_observer import opening_component
    from .hif_navigation_recovery import _verify_retained_opening_source, _verify_unstarted_component_context
    from .hif_native.native_opening_mapping_resume import verify_opening_navigation

    _verify_unstarted_component_context(record, row)
    items, drinks, opening = row['live_events']
    _verify_retained_opening_source(row, previous, proof, prior_reports=[items], counted_return=counted_return)
    confirmed = dict(proof['retained_components'], items=opening_component(
        items, 'items', SimpleNamespace(**row['observed_live_context'])))
    if (opening.get('components') != confirmed or opening.get('unstarted_retained_components') != sorted(confirmed)
            or opening.get('retained_hand_identity_inspection') != previous['live_events'][-1].get('retained_hand_identity_inspection')):
        raise ValueError('opening drink observation changed its completed components')
    verify_opening_navigation(row, [items, drinks], opening, rejected_drink_open=True)
    scan, = drinks['status_events']
    occupancy, inspected = scan['events']
    slots = [item['slot'] for item in occupancy['slots'] if item['state'] == 'OCCUPIED']
    transactions = inspected['transactions']
    details = inspected['items']
    if (not occupancy.get('occupancy_complete') or [s['slot'] for s in occupancy['slots']] != list(range(4))
            or [t['slot'] for t in transactions] != slots[:len(transactions)] or not transactions
            or [d['slot'] for d in details] != [t['slot'] for t in transactions[:-1]]):
        raise ValueError('partial drink readings do not bind the observed physical slots')
    for item, transaction in zip(details, transactions[:-1], strict=True):
        observed = transaction['observation']
        if (item['frame_ids'] != transaction['page_frame_ids']
                or any(item.get(k) != observed.get(k) for k in ('name','confidence','available'))):
            raise ValueError('partial drink identity differs from the completed detail')
    return dict(boundary_kind='unbound_components', source_result=row['id'], input_authorized=False,
        expected_live=proof['expected_live'], initial_core=deepcopy(opening['initial_core']),
        retained_components=confirmed, restore_opening_title=bool(opening.get('retained_hand_identity_inspection')),
        drink_detail_prefix=dict(source_result=row['id'], occupied=slots, items=deepcopy(details)))


def is_open_item_tail(row):
    events = row.get('live_events') or ()
    tail = (row.get('opening_navigation_resume') or {}).get('item_tail_resume') or {}
    if (not tail or len(events) != 2 or events[0].get('event') != 'live_status_observe'
            or events[-1].get('reason') != 'HIF opening component failed: items'):
        return False
    panels = events[0].get('status_events') or ()
    return (len(panels) == 1 and panels[0].get('resumed_item_source') == tail.get('source_result')
            and [item.get('phase') for item in panels[0].get('inputs', ())] == ['open']
            and panels[0].get('after_frame_ids') is None and not panels[0].get('returned_unchanged'))


def resume_open_item_tail(previous, proof, row):
    from .hif_native.native_opening_mapping_resume import verify_opening_navigation

    report, opening = row['live_events']
    panel, = report['status_events']
    tail = proof['item_tail_resume']
    previous_title = previous['published_components'][-1]
    if (opening.get('components') != proof['retained_components'] or panel.get('entered_item_slot') != tail['slot']
            or panel.get('prefix_page_frame_ids') != [page['frame_ids'] for page in tail['pages']]
            or panel.get('pages') or panel.get('complete_list') is not False
            or opening.get('hand_identity_inspection') or opening.get('retained_hand_identity_inspection') != (
                previous_title.get('retained_hand_identity_inspection') or previous_title.get('hand_identity_inspection'))):
        raise ValueError('open item popup differs from its retained opening or slot')
    verify_opening_navigation(row, [report], opening, open_item_tail=True)
    return dict(proof, source_result=row['id'], initial_core=deepcopy(opening['initial_core']),
        item_tail_resume=dict(tail, opened_source_result=row['id'], open_before=deepcopy(opening['initial_core']),
            before_frame_ids=list(panel['before_frame_ids'])))


def is_closed_item_tail_source(row):
    from .hif_item_tail_evidence import SOURCE

    events = row.get('live_events') or ()
    return (row.get('id') == SOURCE and len(events) == 2 and events[0].get('event') == 'live_status_observe'
        and events[1].get('reason') == 'HIF opening component failed: items'
        and row.get('input_count') == 5 and not (row.get('opening_navigation_resume') or {}).get('item_tail_resume'))


def resume_closed_item_tail(previous, proof, row, opening):
    """Extend the already verified card/status chain with its actual item navigation."""
    from .hif_navigation_recovery import verify_unstarted_core
    from .hif_native.native_opening_root import PARENT_FIELDS
    from .hif_native.native_opening_mapping_resume import verify_opening_navigation

    report, published = row['live_events']
    components = proof.get('retained_components', {})
    if (published != opening or set(components) != {'cards', 'status'} or opening.get('components') != components
            or opening.get('unstarted_retained_components') != sorted(components)
            or proof.get('source_result') != previous['id'] or proof.get('input_authorized') is not False
            or any(row.get(k) is not None for k in (*PARENT_FIELDS, 'native_input_preparation', 'native_upstream_input_preparation'))
            or opening.get('extended_hand_identity') or opening.get('hand_identity_inspection')
            or opening.get('retained_hand_identity_inspection') != previous['published_components'][-1].get('retained_hand_identity_inspection')):
        raise ValueError('closed item inspection changed its retained opening prefix')
    panels = report.get('status_events', ())
    if (len(panels) != 1 or report.get('accepted') is not False or report.get('foreground_preparation_events')
            or opening.get('accepted') is not False or not is_closed_item_tail_source(row)):
        raise ValueError('closed item inspection lacks its failed native reader')
    panel = panels[0]
    phases = [item.get('phase') for item in panel.get('inputs', ())]
    if (panel.get('accepted') is not False or panel.get('phase') != 'item_description_inspection'
            or panel.get('popup_confirmed') is not True or panel.get('returned_unchanged') is not True
            or panel.get('complete_list') is not False or panel.get('scroll_drag_pixels') != 80
            or panel.get('read_failure') != 'item description name or limit confidence is insufficient'
            or panel.get('reason') != panel['read_failure'] or phases != ['open', 'scroll', 'scroll', 'scroll', 'close']):
        raise ValueError('closed item inspection is not the completed prefix before the unread item')
    same_host = row.get('host_attempt') is not None and row['host_attempt'] == previous.get('host_attempt')
    previous_inputs = previous['total_input_count'] if same_host else 0
    if row.get('total_input_count') != previous_inputs + row['input_count']:
        raise ValueError('closed item inspection changed its original input count')
    verify_opening_navigation(row, [report], opening, closed_item_failure=True)
    core = opening['initial_core']
    continuity = verify_unstarted_core(proof['initial_core'], core)
    frames = {frame['frame_id'] for frame in row['frames']}
    groups = (continuity['fresh_frame_ids'], panel.get('before_frame_ids', ()), panel.get('after_frame_ids', ()))
    if (opening.get('unstarted_opening_continuity') != continuity
            or any(len(ids) != 3 or len(set(ids)) != 3 or not set(ids) <= frames for ids in groups)
            or set(groups[1]) & set(groups[2])):
        raise ValueError('closed item inspection lacks its original battle return evidence')
    from .hif_item_tail_evidence import verified_item_tail_prefix

    pages = verified_item_tail_prefix(row)
    return dict(proof, source_result=row['id'], initial_core=deepcopy(core),
                item_tail_resume=dict(source_result=row['id'], slot=4, pages=pages))
