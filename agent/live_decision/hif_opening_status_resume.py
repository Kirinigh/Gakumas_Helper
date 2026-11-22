"""Continue a status reader from its observed open panel or completed scroll."""
from copy import deepcopy


def is_open_status_prefix(row):
    events = row.get('live_events') or ()
    if len(events) not in (1, 2) or events[0].get('event') != 'live_status_observe':
        return False
    panels = events[0].get('status_events') or ()
    if len(events) == 1:
        return (len(panels) == 1 and panels[0].get('resumed_status_source')
            and (row.get('opening_navigation_resume') or {}).get('resume_status_panel') is True)
    return (len(panels) == 1 and panels[0].get('phase') == 'status_inspection'
            and [step.get('phase') for step in panels[0].get('inputs', ())] == ['open']
            and set((row.get('opening_navigation_resume') or {}).get('retained_components', {})) == {'cards'})


def is_closed_items_after_status(row):
    events = row.get('live_events') or ()
    return (len(events) == 3 and (row.get('opening_navigation_resume') or {}).get('resume_status_panel') is True
        and events[0].get('event') == 'live_status_observe' and events[0].get('accepted') is True
        and events[1].get('event') == 'live_status_observe' and events[1].get('accepted') is False
        and events[-1].get('reason') == 'HIF opening component failed: items')


def resume_closed_items_after_status(record, previous, proof, row, *, counted_return=False):
    from types import SimpleNamespace

    from .opening_observer import opening_component
    from .hif_item_tail_evidence import verified_closed_item_prefix
    from .hif_navigation_recovery import (
        _verify_retained_opening_source,
        _verify_unstarted_component_context,
    )
    from .hif_native.native_opening_mapping_resume import verify_opening_navigation

    _verify_unstarted_component_context(record, row)
    status, items, opening = row['live_events']
    _verify_retained_opening_source(row, previous, proof, prior_reports=[status], counted_return=counted_return)
    confirmed = dict(proof['retained_components'], status=opening_component(
        status, 'status', SimpleNamespace(**row['observed_live_context'])))
    if (opening.get('components') != confirmed or opening.get('unstarted_retained_components') != sorted(confirmed)
            or not opening.get('retained_hand_identity_inspection') and proof.get('restore_opening_title')):
        raise ValueError('continued opening changed its completed status or card observation')
    verify_opening_navigation(row, [status, items], opening, closed_item_failure=True)
    return dict(boundary_kind='unbound_components', source_result=row['id'], input_authorized=False,
        expected_live=proof['expected_live'], initial_core=deepcopy(opening['initial_core']),
        retained_components=confirmed, restore_opening_title=bool(opening.get('retained_hand_identity_inspection')),
        item_tail_resume=verified_closed_item_prefix(row))


def verify_open_status_source(record, rows, *, parent=None, counted_return=False):
    from .hif_navigation_recovery import (
        verify_unstarted_core,
        _verify_retained_opening_source,
        _verify_closed_notebook_inventory,
        _verify_unstarted_component_context,
    )

    if len(rows) < 2:
        raise ValueError('HIF open status lacks its original inventory')
    previous, row = rows[-2:]
    if parent is None:
        parent = _verify_closed_notebook_inventory(record, rows[:-1])
    _verify_retained_opening_source(row, previous, parent, counted_return=counted_return)
    events = row['live_events']
    if len(events) == 1 and is_open_status_prefix(row):
        report, = events
        panel, = report['status_events']
        original = previous['live_events'][0]['status_events'][0]
        inputs = panel.get('inputs', [])
        if (row.get('status') != 'STOP' or report.get('executed') is not False or report.get('accepted') is not False
                or report.get('execution_submit_count') != 0 or report.get('foreground_preparation_events')
                or row.get('native_input_preparation') or row.get('native_upstream_input_preparation')
                or panel.get('phase') != 'status_inspection' or panel.get('accepted') is not False
                or panel.get('returned_unchanged') or panel.get('after_frame_ids')
                or panel.get('resumed_status_source') != parent.get('status_panel_source', previous['id'])
                or panel.get('before_frame_ids') != original.get('before_frame_ids')
                or panel.get('pages', [])[:len(original['pages'])] != original['pages']
                or any(item.get('phase') != 'scroll' for item in inputs)):
            raise ValueError('HIF continued status lost its original open panel or page prefix')
        _verify_status_inputs(row, report, panel, len(inputs), previous['endpoints']['game']['pid'])
        result = dict(parent, source_result=row['id'])
        if parent.get('restore_opening_title'):
            result['opening_title_source'] = parent.get('opening_title_source', previous['id'])
        if inputs:
            result.pop('status_panel_source', None)
        else:
            result['status_panel_source'] = parent.get('status_panel_source', previous['id'])
        return result
    request, _ = _verify_unstarted_component_context(record, row)
    if [e.get('event') for e in events] != ['live_status_observe', 'live_opening_state']:
        raise ValueError('HIF open status contains another observer')
    report, opening = events
    core = opening.get('initial_core') or {}
    cards = (opening.get('components') or {}).get('cards') or {}
    direct_cards = parent.get('retained_components', {}).get('cards')
    if direct_cards is not None:
        if direct_cards != cards:
            raise ValueError('HIF open status changed its confirmed card table')
    else:
        source_inventory = previous['live_events'][0]['inventory_events'][0]
        if ((cards.get('recovered_inventory') or {}).get('source') != previous['id']
                or any(cards.get(k) != source_inventory.get(k) for k in ('before_frame_ids', 'after_frame_ids'))):
            raise ValueError('HIF open status changed the recovered inventory')
    original = previous['live_events'][-1]
    if (report.get('accepted') is not False or report.get('foreground_preparation_events')
            or len(report.get('status_events', ())) != 1 or opening.get('unstarted_retained_components') != ['cards']
            or set(opening.get('components', {})) != {'cards'} or cards.get('accepted') is not True
            or cards.get('returned_unchanged') is not True or cards.get('input_click_count') != 0
            or any(cards.get(k) != row['observed_live_context'].get(k) for k in ('run_id', 'revision'))
            or opening.get('hand_identity_inspection') is not None
            or opening.get('retained_hand_identity_inspection') != (
                original.get('hand_identity_inspection') or original.get('retained_hand_identity_inspection'))
            or opening.get('unstarted_opening_continuity') != verify_unstarted_core(parent['initial_core'], core)):
        raise ValueError('HIF open status changed the recovered opening')
    panel = report['status_events'][0]
    inputs = panel.get('inputs', [])
    if (panel.get('phase') != 'status_inspection' or panel.get('accepted') is not False
            or panel.get('after_frame_ids') or panel.get('returned_unchanged')
            or [i.get('phase') for i in inputs] != (['open', 'scroll'] if panel.get('pages') else ['open'])
            or row.get('native_upstream_input_preparation') is not None):
        raise ValueError('HIF open status lacks its completed panel navigation')
    _verify_status_inputs(row, report, panel, opening['total_input_click_count'], request.game_pid)
    return dict(boundary_kind='unbound_components', source_result=row['id'], input_authorized=False,
        expected_live=parent['expected_live'], initial_core=deepcopy(core), resume_status_panel=True,
        retained_components={'cards': deepcopy(cards)}, **({'restore_opening_title':True}
            if parent.get('restore_opening_title') else {}))


def _verify_status_inputs(row, report, panel, opening_count, game_pid):
    from .hif_native.native_inprocess_resume import _same_input_host
    from .hif_native.native_settlement_resume import same_endpoint_observation
    from .hif_native.native_pending_components import _navigation_dispatch_matches

    inputs, dispatches = panel.get('inputs', []), row.get('native_input_dispatches', [])
    if any(count != len(inputs) for count in (len(dispatches), row.get('input_count'), row.get('total_input_count'),
            panel.get('input_click_count'), report.get('total_input_click_count'), opening_count)):
        raise ValueError('HIF continued status has unaccounted navigation')
    for item, sent in zip(inputs, dispatches, strict=True):
        endpoints = item.get('endpoints')
        if (item.get('completed') is not True
                or not same_endpoint_observation(sent.get('endpoints'), endpoints, game_pid)
                or not _navigation_dispatch_matches(dict(sent, endpoints=endpoints), item, status=True)):
            raise ValueError('HIF status input differs from its completed navigation')
        _same_input_host(row['endpoints'], sent['endpoints'])
    # The swipe writer keeps its preparation even after SDK completion. Its
    # coordinates, duration and endpoints must belong to the last sent swipe.
    prepared = row.get('native_navigation_preparation') or {}
    if not dispatches:
        if prepared:
            raise ValueError('HIF continued status has an outstanding navigation preparation')
        return
    sent = dispatches[-1]
    if (inputs[-1]['phase'] == 'scroll' and (sent.get('duration') != 400
            or prepared != dict(sent, dispatch_state='prepared_unconfirmed'))
            or inputs[-1]['phase'] == 'open' and (prepared.get('dispatch_state') != 'dispatch_attempted'
                or any(prepared.get(k) != sent.get(k) for k in ('x','y','endpoints')))):
        raise ValueError('HIF status has an unaccounted navigation preparation')


def status_resume_request(source):
    import cv2

    from .inventory import frame_identifier
    from .status_panel import status_panel_geometry

    report, *opening = source['live_events']
    core = opening[0]['initial_core'] if opening else source['opening_navigation_resume']['initial_core']
    request = dict(source_result=source['id'], before=deepcopy(core),
        status=deepcopy(report['status_events'][0]))
    # This historical frame is only needed to distinguish a stalled swipe
    # from an advanced page. The live reader always observes its current page.
    frames = [f for f in source.get('frames', ()) if f.get('label') == 'pre-swipe']
    if frames:
        frame = frames[-1]
        image = cv2.imread(frame['path'])
        if image is not None and frame_identifier(image) == frame['frame_id']:
            request['scroll_anchor'] = dict(frame_id=frame['frame_id'],
                geometry=status_panel_geometry(image, require_complete_layout=False))
    return request
