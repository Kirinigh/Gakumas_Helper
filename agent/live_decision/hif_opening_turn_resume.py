"""Retain a verified opening when the turn-table opening click never reached Maa."""
from copy import deepcopy
from types import SimpleNamespace


def verify_turn_open_rejection(row, report):
    from .hif_native.native_inprocess_resume import _same_input_host
    from .hif_native.native_settlement_resume import same_endpoint_observation
    from .hif_native.native_pending_components import _navigation_dispatch_matches

    panels = report.get('status_events', ())
    if report.get('accepted') is not False or report.get('total_input_click_count') != 1 or len(panels) != 1:
        raise ValueError('HIF turn table lacks a single failed opening attempt')
    turn = panels[0]
    inputs = turn.get('inputs', ())
    reject = row.get('native_navigation_preparation') or {}
    if (turn.get('phase') != 'turn_information_observation' or turn.get('accepted') is not False
            or turn.get('input_click_count') != 1 or len(inputs) != 1 or inputs[0].get('phase') != 'open'
            or inputs[0].get('completed') is True or turn.get('after_frame_ids') or turn.get('returned_unchanged')
            or reject.get('dispatch_state') != 'rejected_before_dispatch'
            or reject.get('input_may_have_been_dispatched') is not False or reject.get('reason') != turn.get('reason')):
        raise ValueError('HIF turn-table input was not definitively rejected before dispatch')
    endpoints = inputs[0].get('endpoints')
    if (not same_endpoint_observation(reject.get('endpoints'), endpoints, row['endpoints']['game']['pid'])
            or not _navigation_dispatch_matches(dict(reject, endpoints=endpoints), inputs[0])):
        raise ValueError('HIF rejected turn-table preparation differs from its target')
    _same_input_host(row['endpoints'], reject['endpoints'])


def verify_retained_turn_source(record, rows):
    from .opening_observer import opening_component
    from .hif_navigation_recovery import _verify_retained_opening_source, _verify_unstarted_component_context
    from .hif_opening_status_resume import verify_open_status_source
    from .hif_native.native_opening_mapping_resume import verify_opening_navigation

    previous, row = rows[-2:]
    parent = verify_open_status_source(record, rows[:-1])
    _verify_unstarted_component_context(record, row)
    events = row['live_events']
    phases = [e['phase'] for report in events[:-1] for e in report.get('status_events', [])]
    if ([e.get('event') for e in events] != ['live_status_observe']*4+['live_opening_state']
            or phases[:1] != ['status_inspection'] or len(phases) != 4
            or phases[1] not in ('item_description_inspection', 'item_hud_inventory')
            or phases[2:] != ['complete_drink_inventory', 'turn_information_observation']):
        raise ValueError('HIF turn-table opening has an unaccounted observer')
    status, items, drinks, turns, opening = events
    _verify_retained_opening_source(row, previous, parent, prior_reports=[status])
    if (status['status_events'][0].get('resumed_status_source') != previous['id']
            or opening.get('unstarted_retained_components') != ['cards', 'status']
            or opening.get('retained_hand_identity_inspection') != previous['live_events'][-1]['retained_hand_identity_inspection']
            or opening.get('hand_identity_inspection') is not None
            or row.get('native_upstream_input_preparation') is not None
            or row.get('total_input_count') != row.get('input_count')):
        raise ValueError('HIF turn-table stop changed its recovered opening')
    run = SimpleNamespace(**row['observed_live_context'])
    components = dict(parent['retained_components'], status=opening_component(status, 'status', run),
        items=opening_component(items, 'items', run), drinks=opening_component(drinks, 'drinks', run))
    if opening.get('components') != components:
        raise ValueError('HIF turn-table stop changed a completed component')
    verify_turn_open_rejection(row, turns)
    verify_opening_navigation(row, events[:3], opening, rejected_turn_open=True)
    return dict(boundary_kind='unbound_components', source_result=row['id'], input_authorized=False,
        expected_live=parent['expected_live'], initial_core=deepcopy(opening['initial_core']),
        retained_components=deepcopy(components), restore_opening_title=True)
